"""M5: same loop as M4, but one iteration now produces one token instead of a whole request.

The scheduler decides who runs and when a request finishes; ModelRunner does the compute.
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
    def __init__(self, model_runner: ModelRunnerLike, eos_token_ids: set[int]):
        self.model_runner = model_runner
        self.eos_token_ids = eos_token_ids

        # Thread-safe inbox: handler threads put, scheduler thread gets.
        self.recv_queue: queue.Queue[Req] = queue.Queue()
        # Scheduler-private FIFO; only the scheduler thread touches it.
        self.waiting_queue: collections.deque[Req] = collections.deque()
        # The request being served, or None when idle. Now lives across many loop iterations.
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
        # TODO(M5-3): capacity is one. A request now spans many iterations:
        # keep the running request, or start the oldest waiting one. None if nothing to run.
        raise NotImplementedError

    def run_batch(self, batch: Req) -> int:
        # TODO(M5-4): one decode step via self.model_runner. Return the next token id.
        raise NotImplementedError

    def process_batch_result(self, batch: Req, result: int | None) -> None:
        # TODO(M5-5): result is the new token, or None if run_batch raised (batch.error is set).
        # 1. Append the token; set finish_reason "stop" (EOS) or "length" (max_new_tokens).
        # 2. Still generating: return and keep running_req.
        # 3. Finished or failed: clear running_req, then set batch.done last.
        raise NotImplementedError
