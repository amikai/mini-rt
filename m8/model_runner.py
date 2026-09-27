"""M8: ModelRunner owns the model, its device, and one forward step for a whole batch (tokens -> logits -> next tokens).

No queue or request-lifecycle logic here; the Scheduler decides who runs and when they finish.
"""

import torch
from transformers import AutoModelForCausalLM

from req import Req
from sampler import Sampler

# Any id works in a padding slot; attention_mask hides it.
PAD_TOKEN_ID = 0


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

    def prepare_inputs(self, batch: list[Req]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-pad every sequence to the longest one.

        Returns input_ids, attention_mask, position_ids, each [batch_size, max_len].
        """
        # No KV cache, so each row is the full sequence every step.
        max_len = max([len(req.input_ids) + len(req.output_ids) for req in batch])
        input_ids = []
        masks = []
        position_ids = []
        for req in batch:
            padding_len = max_len - (len(req.input_ids) + len(req.output_ids))
            # Left padding: every row's last real token lands in the last column.
            row = [PAD_TOKEN_ID] * padding_len
            row += req.input_ids + req.output_ids
            input_ids.append(row)

            masks.append([0] * padding_len + [1] * (max_len - padding_len))
            # Count from each row's first real token, as when it runs alone.
            position_ids.append([0] * padding_len + list(range(max_len - padding_len)))

        return (
            torch.tensor(input_ids, dtype=torch.long, device=self.device),
            torch.tensor(masks, dtype=torch.long, device=self.device),
            torch.tensor(position_ids, dtype=torch.long, device=self.device),
        )

    @torch.inference_mode()
    def forward(self, batch: list[Req]) -> torch.Tensor:
        """Recompute every full sequence (no KV cache). Returns logits of each last token: [batch_size, vocab_size]."""
        input_ids, attention_mask, position_ids = self.prepare_inputs(batch)
        # Without the mask, real tokens would attend to padding.
        outputs = self.model.forward(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        )
        # Left padding puts every last token in the last column.
        return outputs.logits[:, -1, :]

    def sample(self, logits: torch.Tensor, batch: list[Req]) -> list[int]:
        # batch is unused for now; per-request sampling params come later.
        return self.sampler(logits)
