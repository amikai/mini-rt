"""M6: same loop as M5, but every change to a request goes through its state machine.

The scheduler drives status transitions; Req decides when it is finished.
"""

import collections
import queue
from typing import Protocol

import torch

from req import FINISH_ERROR, Req, ReqStatus


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
        # Capacity is one: keep the running request, else start the oldest waiting one.
        if self.running_req is None and self.waiting_queue:
            self.running_req = self.waiting_queue.popleft()
            self.running_req.set_status(ReqStatus.RUNNING)
        return self.running_req

    def run_batch(self, batch: Req) -> int:
        logits = self.model_runner.forward(batch)
        next_token_id = self.model_runner.sample(logits, batch)
        return next_token_id

    def process_batch_result(self, batch: Req, result: int | Exception) -> None:
        # result is the new token, or the exception run_batch raised.
        if isinstance(result, Exception):
            batch.finished_reason = FINISH_ERROR(result)
            batch.set_status(ReqStatus.FAILED)
        else:
            batch.output_ids.append(result)
            batch.update_finish_state()
            if not batch.finished():
                return
            batch.set_status(ReqStatus.FINISHED)
        self.running_req = None
        # Last: the handler may read batch as soon as done is set.
        batch.done.set()
