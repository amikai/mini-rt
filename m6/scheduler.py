"""M6: same loop as M5, but every change to a request goes through its state machine.

The scheduler drives status transitions; Req decides when it is finished.
"""

import collections
import queue
from typing import Protocol

import torch

from req import Req


class ModelRunnerLike(Protocol):
    def forward(self, req: Req) -> torch.Tensor: ...
    def sample(self, logits: torch.Tensor, req: Req) -> int: ...


class Scheduler:
    def __init__(self, model_runner: ModelRunnerLike):
        self.model_runner = model_runner

        # Thread-safe inbox: handler threads put, scheduler thread gets.
        self.recv_queue: queue.Queue[Req] = queue.Queue()
        # Scheduler-private FIFO of WAITING requests; only the scheduler thread touches it.
        self.waiting_queue: collections.deque[Req] = collections.deque()
        # The RUNNING request, or None when idle.
        self.running_req: Req | None = None

    def submit(self, req: Req) -> None:
        """Called from a handler thread. Does not block."""
        self.recv_queue.put(req)

    def event_loop(self) -> None:
        """Runs forever in the scheduler thread."""
        # Each iteration predicts exactly one token for the running request.
        while True:
            self.recv_requests()
            batch = self.get_next_batch_to_run()
            if batch is None:
                continue
            # Fail only this batch on error, so the scheduler thread keeps running.
            try:
                result = self.run_batch(batch)
            except Exception as e:  # noqa: BLE001
                result = e
            self.process_batch_result(batch, result)

    def recv_requests(self) -> None:
        # Fully idle: sleep until the first request arrives instead of spinning.
        if self.running_req is None and not self.waiting_queue:
            self.waiting_queue.append(self.recv_queue.get())
        # Take a snapshot; later arrivals wait for the next step. Only this thread consumes, so get() never blocks.
        for _ in range(self.recv_queue.qsize()):
            self.waiting_queue.append(self.recv_queue.get())

    def get_next_batch_to_run(self) -> Req | None:
        # TODO(M6-3): same as M5, plus the WAITING -> RUNNING transition when a request starts.
        raise NotImplementedError

    def run_batch(self, batch: Req) -> int:
        logits = self.model_runner.forward(batch)
        next_token_id = self.model_runner.sample(logits, batch)
        return next_token_id

    def process_batch_result(self, batch: Req, result: int | Exception) -> None:
        # TODO(M6-4): result is the new token, or the exception run_batch raised.
        # 1. Exception: FINISH_ERROR, then RUNNING -> FAILED.
        # 2. Token: append it and let the request update its finish state.
        #    Not finished: return and keep running_req. Finished: RUNNING -> FINISHED.
        # 3. Ended either way: clear running_req, then set batch.done last.
        raise NotImplementedError
