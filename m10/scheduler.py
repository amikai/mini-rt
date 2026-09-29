"""M10: same loop as M09. Each request's SamplingParams travel with it into the batch.

Continuous batching: an ended request leaves running_batch before the next step, and a waiting request takes its slot.
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
        # Requests of the current batch. Ended ones stay until the next get_next_batch_to_run() drops them.
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
            req = next((r for r in self.running_batch if r.rid == recv_req.rid and not r.finished()), None)
        if req is None:
            return  # already ended, or never existed
        req.finished_reason = FINISH_ABORT()
        # No token this time; the stream still needs to hear that it ended.
        req.out_queue.put((None, req.finished_reason))
        self.end_req(req, ReqStatus.ABORTED)

    def get_next_batch_to_run(self) -> list[Req] | None:
        # Drop requests that ended since the last step, like SGLang's filter_batch().
        self.running_batch = [req for req in self.running_batch if not req.finished()]
        # Fill the free slots, like SGLang's merge_batch().
        self.running_batch.extend(self.get_new_batch_prefill())
        return self.running_batch or None

    def get_new_batch_prefill(self) -> list[Req]:
        # Oldest waiting requests first, up to the free slots.
        free_slots = self.max_batch_size - len(self.running_batch)
        new_reqs = []
        for _ in range(min(free_slots, len(self.waiting_queue))):
            req = self.waiting_queue.popleft()
            req.set_status(ReqStatus.RUNNING)
            new_reqs.append(req)
        return new_reqs

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
        """Every ending goes through here: terminal status, then wake the waiter.

        Caller first sets finished_reason and puts the last out_queue item.
        The request stays in running_batch; get_next_batch_to_run() drops it.
        """
        req.set_status(status)
        # Last: the handler may read req as soon as done is set.
        req.done.set()
