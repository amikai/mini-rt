"""M7: same loop as M6, plus two things per step: stream each new token out, and drop aborted requests.

The scheduler drives status transitions; Req decides when it is finished.
"""

import collections
import queue
from typing import Protocol

import torch

from req import FINISH_ABORT, FINISH_ERROR, AbortReq, Req, ReqStatus


class ModelRunnerLike(Protocol):
    def forward(self, req: Req) -> torch.Tensor: ...
    def sample(self, logits: torch.Tensor, req: Req) -> int: ...


class Scheduler:
    def __init__(self, model_runner: ModelRunnerLike):
        self.model_runner = model_runner

        # Thread-safe inbox: handler threads put, scheduler thread gets.
        self.recv_queue: queue.Queue[Req | AbortReq] = queue.Queue()
        # Scheduler-private FIFO of WAITING requests; only the scheduler thread touches it.
        self.waiting_queue: collections.deque[Req] = collections.deque()
        # The RUNNING request, or None when idle.
        self.running_req: Req | None = None

    def submit(self, recv_req: Req | AbortReq) -> None:
        """Called from a handler thread. Does not block."""
        self.recv_queue.put(recv_req)

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
        # Fully idle: sleep until the first message arrives instead of spinning.
        if self.running_req is None and not self.waiting_queue:
            self.process_input_request(self.recv_queue.get())
        # Take a snapshot; later arrivals wait for the next step. Only this thread consumes, so get() never blocks.
        for _ in range(self.recv_queue.qsize()):
            self.process_input_request(self.recv_queue.get())

    def process_input_request(self, recv_req: Req | AbortReq) -> None:
        # Dispatch by type, like SGLang's process_input_requests().
        if isinstance(recv_req, AbortReq):
            self.abort_request(recv_req)
        else:
            self.waiting_queue.append(recv_req)

    def abort_request(self, recv_req: AbortReq) -> None:
        req = next((r for r in self.waiting_queue if r.rid == recv_req.rid), None)
        if req is not None:
            self.waiting_queue.remove(req)
        elif self.running_req is not None and self.running_req.rid == recv_req.rid:
            req = self.running_req  # end_req frees the running slot
        else:
            return  # already ended, or never existed
        req.finished_reason = FINISH_ABORT()
        # No token this time; the stream still needs to hear that it ended.
        req.out_queue.put((None, req.finished_reason))
        self.end_req(req, ReqStatus.ABORTED)

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
            batch.out_queue.put((None, batch.finished_reason))
            self.end_req(batch, ReqStatus.FAILED)
            return
        batch.output_ids.append(result)
        batch.update_finish_state()
        # After update_finish_state, so the last token carries its finish reason.
        batch.out_queue.put((result, batch.finished_reason))
        if batch.finished():
            self.end_req(batch, ReqStatus.FINISHED)

    def end_req(self, req: Req, status: ReqStatus) -> None:
        """Every ending goes through here: terminal status, free the running slot, wake the waiter.

        Caller first sets finished_reason and puts the last out_queue item.
        """
        req.set_status(status)
        if self.running_req is req:
            self.running_req = None
        # Last: the handler may read req as soon as done is set.
        req.done.set()
