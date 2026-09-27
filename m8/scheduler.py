"""M8: same loop as M7, but each step runs a batch of requests in one forward pass.

Static batching: a new batch starts only after every request in the current one has ended.
The scheduler drives status transitions; Req decides when it is finished.
"""

import collections
import queue
from typing import Protocol

import torch

from req import FINISH_ABORT, FINISH_ERROR, AbortReq, Req, ReqStatus


class ModelRunnerLike(Protocol):
    def forward(self, batch: list[Req]) -> torch.Tensor: ...
    def sample(self, logits: torch.Tensor, batch: list[Req]) -> list[int]: ...


class Scheduler:
    def __init__(self, model_runner: ModelRunnerLike, max_batch_size: int = 8):
        self.model_runner = model_runner
        # Most requests in one forward pass; SGLang's --max-running-requests.
        self.max_batch_size = max_batch_size

        # Thread-safe inbox: handler threads put, scheduler thread gets.
        self.recv_queue: queue.Queue[Req | AbortReq] = queue.Queue()
        # Scheduler-private FIFO of WAITING requests; only the scheduler thread touches it.
        self.waiting_queue: collections.deque[Req] = collections.deque()
        # RUNNING requests of the current batch; empty when idle.
        self.running_batch: list[Req] = []

    def submit(self, recv_req: Req | AbortReq) -> None:
        """Called from a handler thread. Does not block."""
        self.recv_queue.put(recv_req)

    def event_loop(self) -> None:
        """Runs forever in the scheduler thread."""
        # Each iteration predicts exactly one token for every request in the batch.
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
        if not self.running_batch and not self.waiting_queue:
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
        else:
            # end_req takes it out of running_batch.
            req = next((r for r in self.running_batch if r.rid == recv_req.rid), None)
        if req is None:
            return  # already ended, or never existed
        req.finished_reason = FINISH_ABORT()
        # No token this time; the stream still needs to hear that it ended.
        req.out_queue.put((None, req.finished_reason))
        self.end_req(req, ReqStatus.ABORTED)

    def get_next_batch_to_run(self) -> list[Req] | None:
        # Static batching: keep the current batch, even with free slots, until every request in it has ended.
        # Return a copy: end_req() removes from running_batch while process_batch_result() loops over the batch.
        if len(self.running_batch) != 0:
            return list(self.running_batch)

        if not self.waiting_queue:
            return None

        nxt_batch = []
        while len(nxt_batch) < self.max_batch_size and self.waiting_queue:
            batch = self.waiting_queue.popleft()
            batch.set_status(ReqStatus.RUNNING)
            nxt_batch.append(batch)
        self.running_batch = nxt_batch
        return list(self.running_batch)

    def run_batch(self, batch: list[Req]) -> list[int]:
        logits = self.model_runner.forward(batch)
        next_token_ids = self.model_runner.sample(logits, batch)
        return next_token_ids

    def process_batch_result(self, batch: list[Req], result: list[int] | Exception) -> None:
        # One forward pass serves the whole batch, so an error fails every request in it.
        if isinstance(result, Exception):
            for req in batch:
                req.finished_reason = FINISH_ERROR(result)
                req.out_queue.put((None, req.finished_reason))
                self.end_req(req, ReqStatus.FAILED)
            return

        # result[i] is the next token of batch[i].
        for i in range(len(batch)):
            batch[i].output_ids.append(result[i])
            batch[i].update_finish_state()
            # After update_finish_state, so the last token carries its finish reason.
            batch[i].out_queue.put((result[i], batch[i].finished_reason))
            if batch[i].finished():
                self.end_req(batch[i], ReqStatus.FINISHED)

    def end_req(self, req: Req, status: ReqStatus) -> None:
        """Every ending goes through here: terminal status, leave running_batch, wake the waiter.

        Caller first sets finished_reason and puts the last out_queue item.
        """
        req.set_status(status)
        if req in self.running_batch:
            self.running_batch.remove(req)
        # Last: the handler may read req as soon as done is set.
        req.done.set()
