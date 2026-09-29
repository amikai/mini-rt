import threading
import time

import pytest

from req import Req
from scheduler import Scheduler


def make_req(rid: str) -> Req:
    return Req(rid=rid, prompt="", input_ids=[1], max_new_tokens=1)


@pytest.fixture
def started():
    # Start a scheduler thread around a given run_fn; tests swap run_fn freely.
    def start(run_fn):
        scheduler = Scheduler(run_fn=lambda req: run_fn(req))
        threading.Thread(target=scheduler.event_loop, daemon=True).start()
        return scheduler

    return start


def test_requests_run_in_fifo_order(started):
    order = []
    scheduler = started(lambda req: order.append(req.rid))

    reqs = [make_req(rid) for rid in "ABC"]
    for req in reqs:
        scheduler.submit(req)

    assert all(req.done.wait(timeout=5) for req in reqs)
    assert order == ["A", "B", "C"]


def test_one_request_runs_at_a_time(started):
    running = 0
    max_running = 0
    lock = threading.Lock()

    def slow(req):
        nonlocal running, max_running
        with lock:
            running += 1
            max_running = max(max_running, running)
        time.sleep(0.02)
        with lock:
            running -= 1

    scheduler = started(slow)
    reqs = [make_req(str(i)) for i in range(5)]
    for req in reqs:
        scheduler.submit(req)

    assert all(req.done.wait(timeout=5) for req in reqs)
    assert max_running == 1


def test_request_submitted_while_running_waits_its_turn(started):
    # Hold A inside run_fn, submit B, then release A.
    entered = threading.Event()
    release = threading.Event()
    order = []

    def run(req):
        if req.rid == "A":
            entered.set()
            release.wait(timeout=5)
        order.append(req.rid)

    scheduler = started(run)
    a, b = make_req("A"), make_req("B")
    scheduler.submit(a)
    assert entered.wait(timeout=5)
    scheduler.submit(b)

    assert not b.done.wait(timeout=0.1)  # queued, not rejected
    release.set()
    assert b.done.wait(timeout=5)
    assert order == ["A", "B"]


def test_error_fails_only_that_request(started):
    def run(req):
        if req.rid == "bad":
            raise RuntimeError("boom")
        req.output_ids = [0]

    scheduler = started(run)
    bad, good = make_req("bad"), make_req("good")
    scheduler.submit(bad)
    scheduler.submit(good)

    assert bad.done.wait(timeout=5)
    assert isinstance(bad.error, RuntimeError)
    assert good.done.wait(timeout=5)
    assert good.error is None
    assert good.output_ids == [0]


def test_idle_scheduler_does_not_spin(started):
    started(lambda req: None)

    # process_time counts CPU across all threads; a busy loop would burn about 0.5s here.
    start = time.process_time()
    time.sleep(0.5)
    assert time.process_time() - start < 0.1
