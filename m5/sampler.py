"""M5: Sampler turns logits into the next token. Greedy only; per-request settings come later."""

import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor) -> int:
        """logits: [vocab_size] for the last position. Returns one token id."""
        return int(logits.argmax())
