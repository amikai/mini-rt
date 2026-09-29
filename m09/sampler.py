"""M09: Sampler turns a batch of logits into one next token per row. Greedy only; per-request settings come later."""

import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor) -> list[int]:
        """logits: [batch_size, vocab_size], one row per request. Returns one token id per row."""
        return logits.argmax(dim=-1).tolist()
