"""M4 Engine: M3's Engine with the busy lock replaced by a scheduler thread.

generate() queues the request and blocks until the scheduler finishes it. No more EngineBusyError.
"""

import threading
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from req import Req
from scheduler import Scheduler


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Engine:
    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", device: str | None = None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device)
        # Inference mode for layers that act differently in training (Dropout, BatchNorm).
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = {eos} if isinstance(eos, int) else set(eos)

        # Lambda so tests can monkeypatch self._run after construction.
        self.scheduler = Scheduler(run_fn=lambda req: self._run(req))
        # Daemon: the thread dies with the process.
        threading.Thread(target=self.scheduler.event_loop, daemon=True).start()

    def generate(self, prompt: str, max_new_tokens: int = 32) -> Req:
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        req = Req(
            rid=uuid.uuid4().hex,
            prompt=prompt,
            input_ids=self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
        )
        self.scheduler.submit(req)
        req.done.wait()
        if req.error is not None:
            raise req.error
        return req

    def decode(self, req: Req) -> str:
        return self.tokenizer.decode(req.output_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def _run(self, req: Req) -> None:
        # Same model call as M3: HF generate(), model's default sampling, no KV cache.
        input_ids = torch.tensor([req.input_ids], device=self.device)
        generated = self.model.generate(input_ids, max_new_tokens=req.max_new_tokens, do_sample=True, use_cache=False)
        req.output_ids = generated[0, len(req.input_ids) :].tolist()

        # generate() stops at EOS or at max_new_tokens; the last token tells which.
        req.finish_reason = "stop" if req.output_ids[-1] in self.eos_token_ids else "length"
