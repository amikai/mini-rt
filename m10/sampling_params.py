"""M10: SamplingParams says how one request picks its next token. Subset of SGLang's (sampling/sampling_params.py)."""

import math
from dataclasses import dataclass

# A temperature below this counts as 0.
_SAMPLING_EPS = 1e-6
# A top_k that keeps the whole vocabulary.
TOP_K_ALL = 1 << 30


@dataclass
class SamplingParams:
    max_new_tokens: int = 32  # upper bound on generated tokens; EOS may stop earlier
    temperature: float = 1.0  # 0 means greedy; SGLang's default is 1.0 too
    top_p: float = 1.0  # keep the fewest most likely tokens whose probabilities reach top_p
    top_k: int = -1  # keep the top_k most likely tokens; -1 keeps all
    sampling_seed: int | None = None  # same seed, same prompt: same output

    def __post_init__(self):
        # TODO(M10-1): normalize, so the Sampler needs no special case for greedy or "no top_k".
        # 1. temperature 0 (below _SAMPLING_EPS) is greedy: top_k = 1, temperature = 1.0, as SGLang does.
        #    Why 1.0 and not 0: logits are divided by temperature.
        # 2. top_k -1 means the whole vocabulary: top_k = TOP_K_ALL.
        raise NotImplementedError

    def verify(self) -> None:
        """Raises ValueError on a value the Sampler cannot use. Runs after __post_init__."""
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(f"temperature must be a non-negative finite number, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be -1 (disable) or at least 1, got {self.top_k}")
