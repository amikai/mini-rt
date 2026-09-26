"""M7: same loop as M6, plus two things per step: stream each new token out, and drop aborted requests.

The scheduler drives status transitions; Req decides when it is finished.
"""

import collections
import queue
from typing import Protocol

import torch

from req import AbortReq, Req, ReqStatus


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
        # TODO(M7-3): find recv_req.rid in waiting_queue (remove it) or in running_req.
        # Not found: it already ended or never existed; do nothing.
        # Found: FINISH_ABORT, tell the stream there is no token this time, then end it as ABORTED.
        raise NotImplementedError

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
        # TODO(M7-2): same as M6, plus one out_queue item per step, and end_req() for the ending.
        # 1. Exception: FINISH_ERROR, put (None, reason), end as FAILED.
        # 2. Token: append, update finish state, put (token, finished_reason). Finished: end as FINISHED.
        #    Put after update_finish_state, so the last token carries its finish reason.
        raise NotImplementedError

    def end_req(self, req: Req, status: ReqStatus) -> None:
        """Move req to a terminal status and release it. Caller sets finished_reason first."""
        req.set_status(status)
        if self.running_req is req:
            self.running_req = None
        # Last: the handler may read req as soon as done is set.
        req.done.set()
