"""M10: SamplingBatchInfo holds the SamplingParams of a whole batch as per-row tensors.

SGLang's version (sampling/sampling_batch_info.py) lives across steps and is kept in sync with filter_batch()
and merge_batch(). This one is rebuilt every step, like the padded input tensors.
"""

from dataclasses import dataclass

import torch

from req import Req


@dataclass
class SamplingBatchInfo:
    temperatures: torch.Tensor  # [batch_size, 1] float32, so logits / temperatures scales each row
    top_ps: torch.Tensor  # [batch_size] float32
    top_ks: torch.Tensor  # [batch_size] int32
    # [batch_size] int64, or None when no request has a seed; then sampling uses the global RNG.
    sampling_seed: torch.Tensor | None
    is_all_greedy: bool  # every row has top_k 1, so argmax is enough

    @classmethod
    def from_reqs(cls, reqs: list[Req], device: str) -> "SamplingBatchInfo":
        """One row per request, in batch order. SGLang builds it from a ScheduleBatch, which comes later."""
        # TODO(M10-2): read each req.sampling_params into the tensors above, on `device`.
        # - sampling_seed: None if no request has a seed. Otherwise give each unseeded row a fresh random seed,
        #   so every row takes the seeded path but unseeded rows still change from run to run.
        # - is_all_greedy: every top_k <= 1.
        raise NotImplementedError
