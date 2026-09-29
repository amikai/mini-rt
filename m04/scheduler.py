"""M4: scheduler loop.

Requests queue up instead of being rejected. One request runs at a time, in FIFO order.
The scheduler decides who runs; it never touches the model directly.
"""

import collections
import queue
from collections.abc import Callable

from req import Req


class Scheduler:
    def __init__(self, run_fn: Callable[[Req], None]):
        # Compute step, injected so the scheduler has no model or torch logic.
        self.run_fn = run_fn

        # Thread-safe inbox: handler threads put, scheduler thread gets.
        self.recv_queue: queue.Queue[Req] = queue.Queue()
        # Scheduler-private FIFO; only the scheduler thread touches it.
        self.waiting_queue: collections.deque[Req] = collections.deque()
        # The request being served, or None when idle.
        self.running_req: Req | None = None

    def submit(self, req: Req) -> None:
        """Called from a handler thread. Does not block."""
        self.recv_queue.put(req)

    def event_loop(self) -> None:
        """Runs forever in the scheduler thread."""
        while True:
            self.recv_requests()
            batch = self.get_next_batch_to_run()
            if batch is None:
                continue
            # Fail only this batch on error, so the scheduler thread keeps running.
            try:
                result = self.run_batch(batch)
            except Exception as e:  # noqa: BLE001
                batch.error = e
                result = None
            self.process_batch_result(batch, result)

    def recv_requests(self) -> None:
        # Fully idle: sleep until the first request arrives instead of spinning.
        if self.running_req is None and not self.waiting_queue:
            self.waiting_queue.append(self.recv_queue.get())
        # Take a snapshot; later arrivals wait for the next step. Only this thread consumes, so get() never blocks.
        for _ in range(self.recv_queue.qsize()):
            self.waiting_queue.append(self.recv_queue.get())

    def get_next_batch_to_run(self) -> Req | None:
        # Capacity is one: start the oldest waiting request only when nothing is running.
        if self.running_req is not None or not self.waiting_queue:
            return None
        self.running_req = self.waiting_queue.popleft()
        return self.running_req

    def run_batch(self, batch: Req) -> None:
        # run_fn writes output onto the Req, so there is no separate result yet.
        self.run_fn(batch)

    def process_batch_result(self, batch: Req, result: None) -> None:
        self.running_req = None
        # Last: the handler may read batch as soon as it wakes.
        batch.done.set()
