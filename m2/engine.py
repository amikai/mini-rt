"""M2: single request runtime.

Turn `generate(prompt)` into `Req -> Engine -> Model`.
The Engine runs one request at a time and rejects new requests while busy.
"""

import threading
import uuid
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Req:
    """One generation request and everything the runtime knows about it."""

    rid: str
    prompt: str
    input_ids: list[int]
    max_new_tokens: int
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None  # None while running, then "stop" or "length"


class EngineBusyError(Exception):
    """Raised when a request arrives while another request is running."""


class Engine:
    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", device: str | None = None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device)
        # Inference mode for layers that act differently in training (Dropout, BatchNorm).
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = {eos} if isinstance(eos, int) else set(eos)

        # Runtime state: the request being served, or None when idle.
        self.running_req: Req | None = None
        # Guards admission. Held for the whole request, so holding it means "busy".
        self._lock = threading.Lock()

    def generate(self, prompt: str, max_new_tokens: int = 32) -> Req:
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        # Admission: capacity is one request.
        # A plain `if self.running_req is not None` check is not atomic: two threads can both see "idle".
        # acquire(blocking=False) checks and takes the lock in one step, and rejects instead of waiting.
        if not self._lock.acquire(blocking=False):
            raise EngineBusyError("engine is busy with another request")

        try:
            req = Req(
                rid=uuid.uuid4().hex,
                prompt=prompt,
                input_ids=self.tokenizer.encode(prompt),
                max_new_tokens=max_new_tokens,
            )
            self.running_req = req
            self._run(req)
        finally:
            self.running_req = None
            self._lock.release()
        return req

    def decode(self, req: Req) -> str:
        return self.tokenizer.decode(req.output_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def _run(self, req: Req) -> None:
        # The model call is HF model.generate(); the runtime only wraps it with request state.
        # generate() wraps M1's loop: forward -> last logits -> pick token -> append -> stop at EOS or max_new_tokens.
        # Sample with the model's defaults from generation_config (Qwen3: temperature 0.6, top_k 20, top_p 0.95).
        # Per-request sampling settings arrive in M9.
        # No KV cache: recompute the full sequence every step, like M1. The cache arrives in M12.
        input_ids = torch.tensor([req.input_ids], device=self.device)
        generated = self.model.generate(
            input_ids, max_new_tokens=req.max_new_tokens, do_sample=True, use_cache=False
        )
        req.output_ids = generated[0, len(req.input_ids):].tolist()

        # generate() stops at EOS or at max_new_tokens; the last token tells which.
        req.finish_reason = "stop" if req.output_ids[-1] in self.eos_token_ids else "length"
