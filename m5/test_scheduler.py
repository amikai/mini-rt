import threading
import time

import pytest

from req import Req
from scheduler import Scheduler

EOS = 99


class FakeModelRunner:
    """Emits scripted tokens per request; no model, no torch."""

    def __init__(self, scripts=None, on_forward=None):
        self.scripts = scripts or {}
        self.on_forward = on_forward or (lambda req: None)

    def forward(self, req):
        self.on_forward(req)

    def sample(self, logits, req):
        script = self.scripts.get(req.rid, [])
        step = len(req.output_ids)
        return script[step] if step < len(script) else 0


def make_req(rid: str, max_new_tokens: int = 1) -> Req:
    return Req(rid=rid, prompt="", input_ids=[1], max_new_tokens=max_new_tokens)


@pytest.fixture
def started():
    def start(runner):
        scheduler = Scheduler(runner, eos_token_ids={EOS})
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
    assert req.finish_reason == "length"
    assert steps == ["A", "A", "A"]


def test_eos_stops_early(started):
    scheduler = started(FakeModelRunner(scripts={"A": [5, EOS, 7]}))

    req = make_req("A", max_new_tokens=10)
    scheduler.submit(req)

    assert req.done.wait(timeout=5)
    assert req.output_ids == [5, EOS]
    assert req.finish_reason == "stop"


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
    assert isinstance(bad.error, RuntimeError)
    assert good.done.wait(timeout=5)
    assert good.error is None
    assert good.output_ids == [3]


def test_idle_scheduler_does_not_spin(started):
    started(FakeModelRunner())

    # process_time counts CPU across all threads; a busy loop would burn about 0.5s here.
    start = time.process_time()
    time.sleep(0.5)
    assert time.process_time() - start < 0.1
