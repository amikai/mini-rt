"""M5: Sampler turns logits into the next token. Greedy only; per-request settings arrive in M9."""

import torch
from torch import nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor) -> int:
        """logits: [vocab_size] for the last position. Returns one token id."""
        # TODO(M5-1): greedy. Pick the highest-scoring token and return it as a Python int.
        raise NotImplementedError
