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
        self.on_forward = on_forward or (lambda batch: None)

    def forward(self, batch):
        self.on_forward(batch)
        # Scheduler must mark every request RUNNING before the forward step.
        assert all(req.status is ReqStatus.RUNNING for req in batch)

    def sample(self, logits, batch):
        return [self.next_token(req) for req in batch]

    def next_token(self, req):
        script = self.scripts.get(req.rid, [])
        step = len(req.output_ids)
        return script[step] if step < len(script) else 0


def rids(batch) -> list[str]:
    return [req.rid for req in batch]


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
    def start(runner, max_batch_size=8, reqs=()):
        # reqs are submitted before the thread starts, so the first step sees all of them.
        scheduler = Scheduler(runner, max_batch_size=max_batch_size)
        for req in reqs:
            scheduler.submit(req)
        threading.Thread(target=scheduler.event_loop, daemon=True).start()
        return scheduler

    return start


def test_one_token_per_step_until_length(started):
    steps = []
    req = make_req("A", max_new_tokens=3)
    started(FakeModelRunner(scripts={"A": [5, 6, 7]}, on_forward=lambda b: steps.append(rids(b))), reqs=[req])

    assert req.done.wait(timeout=5)
    assert req.output_ids == [5, 6, 7]
    assert req.status is ReqStatus.FINISHED
    assert isinstance(req.finished_reason, FINISH_LENGTH)
    assert steps == [["A"], ["A"], ["A"]]


@pytest.mark.parametrize("eos", [EOS, EOS2])
def test_any_eos_stops_early(started, eos):
    req = make_req("A", max_new_tokens=10)
    started(FakeModelRunner(scripts={"A": [5, eos, 7]}), reqs=[req])

    assert req.done.wait(timeout=5)
    assert req.output_ids == [5, eos]
    assert req.status is ReqStatus.FINISHED
    assert req.finished_reason.to_json() == {"type": "stop", "matched": eos}


def test_waiting_requests_share_one_forward(started):
    steps = []
    reqs = [make_req(rid, max_new_tokens=2) for rid in "ABC"]
    started(FakeModelRunner(on_forward=lambda b: steps.append(rids(b))), reqs=reqs)

    assert all(req.done.wait(timeout=5) for req in reqs)
    assert steps == [["A", "B", "C"], ["A", "B", "C"]]


def test_batch_is_capped_at_max_batch_size_in_fifo_order(started):
    steps = []
    reqs = [make_req(rid) for rid in "ABCDE"]
    started(FakeModelRunner(on_forward=lambda b: steps.append(rids(b))), max_batch_size=2, reqs=reqs)

    assert all(req.done.wait(timeout=5) for req in reqs)
    assert steps == [["A", "B"], ["C", "D"], ["E"]]


def test_each_request_gets_its_own_token(started):
    a, b = make_req("A", max_new_tokens=2), make_req("B", max_new_tokens=2)
    started(FakeModelRunner(scripts={"A": [5, 6], "B": [7, 8]}), reqs=[a, b])

    assert a.done.wait(timeout=5)
    assert b.done.wait(timeout=5)
    assert a.output_ids == [5, 6]
    assert b.output_ids == [7, 8]
    assert drain(a) == [(5, None), (6, FINISH_LENGTH)]
    assert drain(b) == [(7, None), (8, FINISH_LENGTH)]


def test_finished_slot_is_refilled_on_the_next_step(started):
    # Continuous batching: C takes A's slot right after A finishes, while B keeps running.
    steps = []
    a, b, c = make_req("A", max_new_tokens=1), make_req("B", max_new_tokens=3), make_req("C")
    started(FakeModelRunner(on_forward=lambda b: steps.append(rids(b))), max_batch_size=2, reqs=[a, b, c])

    assert all(req.done.wait(timeout=5) for req in (a, b, c))
    assert steps == [["A", "B"], ["B", "C"], ["B"]]


def test_many_slots_free_up_and_refill_in_fifo_order(started):
    # A B C -> _ B C -> D B C; kept requests stay in front, newcomers join at the end.
    steps = []
    reqs = [make_req("A", 1), make_req("B", 3), make_req("C", 3), make_req("D", 2), make_req("E", 1)]
    started(FakeModelRunner(on_forward=lambda b: steps.append(rids(b))), max_batch_size=3, reqs=reqs)

    assert all(req.done.wait(timeout=5) for req in reqs)
    assert steps == [["A", "B", "C"], ["B", "C", "D"], ["B", "C", "D"], ["E"]]


def test_get_next_batch_to_run_drops_ended_and_fills_free_slots():
    # No thread: drive one step by hand.
    scheduler = Scheduler(FakeModelRunner(), max_batch_size=3)
    done, running = make_req("done"), make_req("running")
    for req in (done, running):
        req.set_status(ReqStatus.RUNNING)
    # What end_req() leaves behind.
    done.finished_reason = FINISH_LENGTH(1)
    done.set_status(ReqStatus.FINISHED)
    scheduler.running_batch = [done, running]
    scheduler.waiting_queue.extend([make_req("W1"), make_req("W2"), make_req("W3")])

    batch = scheduler.get_next_batch_to_run()

    assert rids(batch) == ["running", "W1", "W2"]
    assert rids(scheduler.running_batch) == ["running", "W1", "W2"]
    assert all(req.status is ReqStatus.RUNNING for req in batch)
    assert rids(scheduler.waiting_queue) == ["W3"]


def test_get_next_batch_to_run_returns_none_when_everything_ended():
    scheduler = Scheduler(FakeModelRunner())
    req = make_req("A")
    req.set_status(ReqStatus.RUNNING)
    req.finished_reason = FINISH_LENGTH(1)
    req.set_status(ReqStatus.FINISHED)
    scheduler.running_batch = [req]

    assert scheduler.get_next_batch_to_run() is None
    assert scheduler.running_batch == []


def test_request_submitted_while_batch_runs_joins_on_the_next_step(started):
    # Hold A inside forward, submit B, then release A. B does not wait for A to finish.
    entered = threading.Event()
    release = threading.Event()
    steps = []

    def hold_a(batch):
        steps.append(rids(batch))
        if "A" in rids(batch):
            entered.set()
            release.wait(timeout=5)

    a, b = make_req("A", max_new_tokens=2), make_req("B")
    scheduler = started(FakeModelRunner(on_forward=hold_a), reqs=[a])
    assert entered.wait(timeout=5)
    scheduler.submit(b)

    assert not b.done.wait(timeout=0.1)  # queued, not rejected
    assert a.status is ReqStatus.RUNNING
    assert b.status is ReqStatus.WAITING
    release.set()
    assert a.done.wait(timeout=5)
    assert b.done.wait(timeout=5)
    assert steps == [["A"], ["A", "B"]]


def test_error_fails_the_whole_batch_and_only_that_batch(started):
    def boom(batch):
        if "bad" in rids(batch):
            raise RuntimeError("boom")

    bad, mate, later = make_req("bad", max_new_tokens=5), make_req("mate", max_new_tokens=5), make_req("later")
    started(FakeModelRunner(scripts={"later": [3]}, on_forward=boom), max_batch_size=2, reqs=[bad, mate, later])

    for req in (bad, mate):
        assert req.done.wait(timeout=5)
        assert req.status is ReqStatus.FAILED
        assert isinstance(req.finished_reason, FINISH_ERROR)
        assert isinstance(req.finished_reason.error, RuntimeError)
        assert drain(req) == [(None, FINISH_ERROR)]
    assert later.done.wait(timeout=5)
    assert later.status is ReqStatus.FINISHED
    assert later.output_ids == [3]


def test_idle_scheduler_does_not_spin(started):
    started(FakeModelRunner())

    # process_time counts CPU across all threads; a busy loop would burn about 0.5s here.
    start = time.process_time()
    time.sleep(0.5)
    assert time.process_time() - start < 0.1


def test_abort_one_running_request_keeps_the_rest_of_the_batch(started):
    steps = []

    def abort_a_on_third_step(batch):
        steps.append(rids(batch))
        if len(steps) == 3:
            scheduler.submit(AbortReq("A"))

    # Not started(): the callback needs `scheduler` bound before the thread runs.
    scheduler = Scheduler(FakeModelRunner(on_forward=abort_a_on_third_step))
    a, b = make_req("A", max_new_tokens=100), make_req("B", max_new_tokens=5)
    scheduler.submit(a)
    scheduler.submit(b)
    threading.Thread(target=scheduler.event_loop, daemon=True).start()

    assert a.done.wait(timeout=5)
    assert b.done.wait(timeout=5)
    assert a.status is ReqStatus.ABORTED
    assert isinstance(a.finished_reason, FINISH_ABORT)
    assert a.output_ids == [0, 0, 0]
    assert drain(a) == [(0, None), (0, None), (0, None), (None, FINISH_ABORT)]
    assert b.status is ReqStatus.FINISHED
    assert b.output_ids == [0] * 5
    assert steps == [["A", "B"]] * 3 + [["B"]] * 2


def test_abort_waiting_request_removes_it_before_it_runs(started):
    # Hold A inside forward, queue B and its abort, then release A.
    entered = threading.Event()
    release = threading.Event()
    steps = []

    def hold_a(batch):
        steps.append(rids(batch))
        if "A" in rids(batch):
            entered.set()
            release.wait(timeout=5)

    a, b = make_req("A"), make_req("B")
    scheduler = started(FakeModelRunner(on_forward=hold_a), reqs=[a])
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
    assert steps == [["A"]]


def test_abort_arriving_right_after_finish_is_ignored():
    # The abort lands while A has ended but is still in running_batch.
    steps = []

    def abort_a_on_its_last_step(batch):
        steps.append(rids(batch))
        if len(steps) == 1:
            scheduler.submit(AbortReq("A"))

    scheduler = Scheduler(FakeModelRunner(on_forward=abort_a_on_its_last_step))
    a, b = make_req("A", max_new_tokens=1), make_req("B", max_new_tokens=2)
    scheduler.submit(a)
    scheduler.submit(b)
    threading.Thread(target=scheduler.event_loop, daemon=True).start()

    assert a.done.wait(timeout=5)
    assert b.done.wait(timeout=5)
    assert a.status is ReqStatus.FINISHED
    assert drain(a) == [(0, FINISH_LENGTH)]
    assert b.status is ReqStatus.FINISHED
    assert steps == [["A", "B"], ["B"]]


def test_abort_after_finish_or_unknown_rid_is_ignored(started):
    a = make_req("A")
    scheduler = started(FakeModelRunner(), reqs=[a])
    assert a.done.wait(timeout=5)

    scheduler.submit(AbortReq("A"))
    scheduler.submit(AbortReq("nope"))
    b = make_req("B")
    scheduler.submit(b)

    assert b.done.wait(timeout=5)
    assert a.status is ReqStatus.FINISHED
    assert b.status is ReqStatus.FINISHED
