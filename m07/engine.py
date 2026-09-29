"""M7 Engine: wires tokenizer, Scheduler, and ModelRunner together. No model or queue logic of its own.

Engine
  ├─ Scheduler       decides who runs
  └─ ModelRunner     owns device placement and forward execution (tensor -> logits)
       └─ Sampler    turns logits into the next token (greedy)
"""

import threading
import uuid

from transformers import AutoTokenizer

from model_runner import ModelRunner, pick_device
from req import FINISH_ERROR, AbortReq, Req, ReqStatus
from scheduler import Scheduler


class Engine:
    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_runner = ModelRunner(model_id, device or pick_device())
        self.scheduler = Scheduler(self.model_runner)
        # Daemon: the thread dies with the process.
        threading.Thread(target=self.scheduler.event_loop, daemon=True).start()

    def submit(self, prompt: str, max_new_tokens: int = 32) -> Req:
        """Queue a request and return at once. Read tokens from req.out_queue, or wait on req.done."""
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        req = Req(
            rid=uuid.uuid4().hex,
            prompt=prompt,
            input_ids=self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            eos_token_ids=self.model_runner.eos_token_ids,
        )
        self.scheduler.submit(req)
        return req

    def generate(self, prompt: str, max_new_tokens: int = 32) -> Req:
        """Blocking: returns when the request ends."""
        req = self.submit(prompt, max_new_tokens)
        req.done.wait()
        if req.status is ReqStatus.FAILED:
            assert isinstance(req.finished_reason, FINISH_ERROR)
            raise req.finished_reason.error
        return req

    def abort(self, req: Req) -> None:
        # Safe to call after req ended; the scheduler ignores unknown rids.
        self.scheduler.submit(AbortReq(req.rid))

    def decode(self, req: Req) -> str:
        return self.tokenizer.decode(req.output_ids, skip_special_tokens=True)
