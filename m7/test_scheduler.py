import threading
import time

import pytest

from req import FINISH_ABORT, FINISH_ERROR, FINISH_LENGTH, AbortReq, Req, ReqStatus
from scheduler import Scheduler

EOS, EOS2 = 99, 98


class FakeModelRunner:
    """Emits scripted tokens per request; no model, no torch."""

    def __init__(self, scripts=None, on_forward=None):
        self.scripts = scripts or {}
        self.on_forward = on_forward or (lambda req: None)

    def forward(self, req):
        self.on_forward(req)
        # Scheduler must mark the request RUNNING before the forward step.
        assert req.status is ReqStatus.RUNNING

    def sample(self, logits, req):
        script = self.scripts.get(req.rid, [])
        step = len(req.output_ids)
        return script[step] if step < len(script) else 0


def drain(req: Req) -> list[tuple]:
    """Everything the scheduler put on out_queue, as (token_id, finish_reason type or None)."""
    items = []
    while not req.out_queue.empty():
        token_id, reason = req.out_queue.get()
        items.append((token_id, type(reason) if reason else None))
    return items


def make_req(rid: str, max_new_tokens: int = 1) -> Req:
    return Req(rid=rid, prompt="", input_ids=[1], max_new_tokens=max_new_tokens, eos_token_ids={EOS, EOS2})


@pytest.fixture
def started():
    def start(runner):
        scheduler = Scheduler(runner)
        threading.Thread(target=scheduler.event_loop, daemon=True).start()
        return scheduler

    return start


def test_one_token_per_step_until_length(started):
    steps = []
    scheduler = started(FakeModelRunner(scripts={"A": [5, 6, 7]}, on_forward=lambda req: steps.append(req.rid)))

    req = make_req("A", max_new_tokens=3)
    scheduler.submit(req)

    assert req.done.wait(timeout=5)
    assert req.output_ids == [5, 6, 7]
    assert req.status is ReqStatus.FINISHED
    assert isinstance(req.finished_reason, FINISH_LENGTH)
    assert steps == ["A", "A", "A"]


@pytest.mark.parametrize("eos", [EOS, EOS2])
def test_any_eos_stops_early(started, eos):
    scheduler = started(FakeModelRunner(scripts={"A": [5, eos, 7]}))

    req = make_req("A", max_new_tokens=10)
    scheduler.submit(req)

    assert req.done.wait(timeout=5)
    assert req.output_ids == [5, eos]
    assert req.status is ReqStatus.FINISHED
    assert req.finished_reason.to_json() == {"type": "stop", "matched": eos}


def test_requests_run_in_fifo_order_without_interleaving(started):
    steps = []
    scheduler = started(FakeModelRunner(on_forward=lambda req: steps.append(req.rid)))

    reqs = [make_req(rid, max_new_tokens=2) for rid in "ABC"]
    for req in reqs:
        scheduler.submit(req)

    assert all(req.done.wait(timeout=5) for req in reqs)
    assert steps == ["A", "A", "B", "B", "C", "C"]


def test_request_submitted_while_running_waits_its_turn(started):
    # Hold A inside forward, submit B, then release A.
    entered = threading.Event()
    release = threading.Event()

    def hold_a(req):
        if req.rid == "A":
            entered.set()
            release.wait(timeout=5)

    scheduler = started(FakeModelRunner(on_forward=hold_a))
    a, b = make_req("A"), make_req("B")
    scheduler.submit(a)
    assert entered.wait(timeout=5)
    scheduler.submit(b)

    assert not b.done.wait(timeout=0.1)  # queued, not rejected
    assert a.status is ReqStatus.RUNNING
    assert b.status is ReqStatus.WAITING
    release.set()
    assert a.done.wait(timeout=5)
    assert b.done.wait(timeout=5)


def test_error_fails_only_that_request(started):
    def boom(req):
        if req.rid == "bad":
            raise RuntimeError("boom")

    scheduler = started(FakeModelRunner(scripts={"good": [3]}, on_forward=boom))
    bad, good = make_req("bad", max_new_tokens=5), make_req("good")
    scheduler.submit(bad)
    scheduler.submit(good)

    assert bad.done.wait(timeout=5)
    assert bad.status is ReqStatus.FAILED
    assert isinstance(bad.finished_reason, FINISH_ERROR)
    assert isinstance(bad.finished_reason.error, RuntimeError)
    assert good.done.wait(timeout=5)
    assert good.status is ReqStatus.FINISHED
    assert good.output_ids == [3]


def test_idle_scheduler_does_not_spin(started):
    started(FakeModelRunner())

    # process_time counts CPU across all threads; a busy loop would burn about 0.5s here.
    start = time.process_time()
    time.sleep(0.5)
    assert time.process_time() - start < 0.1


def test_each_step_puts_its_token_on_out_queue(started):
    scheduler = started(FakeModelRunner(scripts={"A": [5, 6, 7]}))

    req = make_req("A", max_new_tokens=3)
    scheduler.submit(req)

    assert req.done.wait(timeout=5)
    # The last token carries the finish reason; nothing comes after it.
    assert drain(req) == [(5, None), (6, None), (7, FINISH_LENGTH)]


def test_error_puts_finish_reason_without_token(started):
    def boom(req):
        raise RuntimeError("boom")

    scheduler = started(FakeModelRunner(on_forward=boom))
    req = make_req("A", max_new_tokens=5)
    scheduler.submit(req)

    assert req.done.wait(timeout=5)
    assert drain(req) == [(None, FINISH_ERROR)]


def test_abort_running_request_stops_it_at_next_step(started):
    steps = []

    def abort_a_on_third_step(req):
        steps.append(req.rid)
        if req.rid == "A" and len(req.output_ids) == 2:
            scheduler.submit(AbortReq("A"))

    scheduler = started(FakeModelRunner(on_forward=abort_a_on_third_step))
    a, b = make_req("A", max_new_tokens=100), make_req("B", max_new_tokens=1)
    scheduler.submit(a)
    scheduler.submit(b)

    assert a.done.wait(timeout=5)
    assert a.status is ReqStatus.ABORTED
    assert isinstance(a.finished_reason, FINISH_ABORT)
    assert a.output_ids == [0, 0, 0]
    assert drain(a) == [(0, None), (0, None), (0, None), (None, FINISH_ABORT)]
    # The running slot is free again.
    assert b.done.wait(timeout=5)
    assert steps == ["A", "A", "A", "B"]


def test_abort_waiting_request_removes_it_before_it_runs(started):
    # Hold A inside forward, queue B and its abort, then release A.
    entered = threading.Event()
    release = threading.Event()
    steps = []

    def hold_a(req):
        steps.append(req.rid)
        if req.rid == "A":
            entered.set()
            release.wait(timeout=5)

    scheduler = started(FakeModelRunner(on_forward=hold_a))
    a, b = make_req("A"), make_req("B")
    scheduler.submit(a)
    assert entered.wait(timeout=5)
    scheduler.submit(b)
    scheduler.submit(AbortReq("B"))
    release.set()

    assert b.done.wait(timeout=5)
    assert b.status is ReqStatus.ABORTED
    assert b.output_ids == []
    assert drain(b) == [(None, FINISH_ABORT)]
    assert a.done.wait(timeout=5)
    assert a.status is ReqStatus.FINISHED
    assert steps == ["A"]


def test_abort_after_finish_or_unknown_rid_is_ignored(started):
    scheduler = started(FakeModelRunner())
    a = make_req("A")
    scheduler.submit(a)
    assert a.done.wait(timeout=5)

    scheduler.submit(AbortReq("A"))
    scheduler.submit(AbortReq("nope"))
    b = make_req("B")
    scheduler.submit(b)

    assert b.done.wait(timeout=5)
    assert a.status is ReqStatus.FINISHED
    assert b.status is ReqStatus.FINISHED
