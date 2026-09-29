"""M10: Sampler turns a batch of logits into one next token per row, each row with its own SamplingParams.

Batched tensor ops, no Python loop over requests. Function names follow SGLang's torch path (layers/sampler.py).
"""

import torch
from torch import nn

from sampling_batch_info import SamplingBatchInfo


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, sampling_info: SamplingBatchInfo, positions: torch.Tensor) -> list[int]:
        """logits: [batch_size, vocab_size]. positions: [batch_size], position of each row's last token.

        Returns one token id per row.
        """
        # TODO(M10-3):
        # 1. is_all_greedy: argmax, same as M9.
        # 2. Otherwise, in float32: divide by temperatures, then softmax into probs.
        # 3. apply_top_k_top_p, then sampling_from_probs_torch.
        # A greedy row needs no branch here: top_k 1 leaves one token to sample.
        raise NotImplementedError


def apply_top_k_top_p(probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Zero every token that its row's top_k or top_p drops. Same shape and vocab order; not renormalized."""
    # TODO(M10-4):
    # 1. Sort each row descending. stable=True, so tied tokens keep the lowest id first, like argmax.
    # 2. Drop sorted position >= top_k.
    # 3. Drop a token when the tokens before it already sum to more than top_p. The top token always stays.
    # 4. Put the kept probs back in vocab order (scatter with the sort indices).
    raise NotImplementedError


def sampling_from_probs_torch(
    probs: torch.Tensor, sampling_seed: torch.Tensor | None, positions: torch.Tensor
) -> torch.Tensor:
    """probs: [batch_size, vocab_size], rows need not sum to 1. Returns [batch_size] token ids."""
    if sampling_seed is None:
        return torch.multinomial(probs, num_samples=1).view(-1)
    # log(0) = -inf, so a dropped token never wins.
    return multinomial_with_seed(torch.log(probs), sampling_seed, positions).view(-1)


def multinomial_with_seed(logprobs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """One index per row, drawn from softmax(logprobs). Returns [batch_size, 1].

    The same (seed, position) picks the same token every time, whatever else is in the batch.
    """
    # TODO(M10-5): Gumbel-max trick: argmax(logprobs + g), g = -log(-log(u)), u uniform in (0, 1).
    # - Each row's u depends only on its (seed, position), never on batch mates or the global RNG:
    #   one torch.Generator per row, seeded from both. SGLang hashes them on the device instead.
    # - Clamp u away from 0, or g = -inf can hide the only kept token.
    raise NotImplementedError
