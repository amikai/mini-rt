"""M6: ModelRunner owns the model, its device, and one forward step (tokens -> logits -> next token).

No queue or request-lifecycle logic here; the Scheduler decides who runs and when they finish.
"""

import torch
from transformers import AutoModelForCausalLM

from req import Req
from sampler import Sampler


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelRunner:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
        # Inference mode for layers that act differently in training (Dropout, BatchNorm).
        self.model.eval()
        self.sampler = Sampler()

        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = {eos} if isinstance(eos, int) else set(eos)

    @torch.inference_mode()
    def forward(self, req: Req) -> torch.Tensor:
        """Recompute the full sequence (no KV cache) and return logits for the last position: [vocab_size]."""
        input_ids = torch.tensor([req.input_ids + req.output_ids], device=self.device)
        outputs = self.model(input_ids, use_cache=False)
        return outputs.logits[0, -1]

    def sample(self, logits: torch.Tensor, req: Req) -> int:
        # req is unused for now; per-request sampling params come later.
        return self.sampler(logits)
