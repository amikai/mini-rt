import collections

import pytest
import torch

from req import Req
from sampler import Sampler, apply_top_k_top_p, multinomial_with_seed
from sampling_batch_info import SamplingBatchInfo
from sampling_params import SamplingParams


def greedy() -> SamplingParams:
    return SamplingParams(temperature=0.0)


def sample(logits: torch.Tensor, params: list[SamplingParams], positions: list[int] | None = None) -> list[int]:
    reqs = [Req(rid=str(i), prompt="", input_ids=[1], sampling_params=p) for i, p in enumerate(params)]
    info = SamplingBatchInfo.from_reqs(reqs, "cpu")
    return Sampler()(logits, info, torch.tensor(positions or [0] * len(params)))


def counts(logits: torch.Tensor, params: SamplingParams, n: int = 2000) -> collections.Counter:
    """Sample one row n times in one batch, so each draw is a separate row."""
    return collections.Counter(sample(logits.repeat(n, 1), [params] * n, list(range(n))))


def test_all_greedy_picks_argmax_per_row():
    logits = torch.tensor([[0.1, 2.0, -1.0, 1.9], [3.0, 0.0, 0.0, 0.0]])

    assert sample(logits, [greedy(), greedy()]) == [1, 0]


def test_returns_python_ints():
    tokens = sample(torch.randn(3, 10), [greedy(), SamplingParams(), SamplingParams(sampling_seed=1)])

    assert all(type(t) is int for t in tokens)


def test_greedy_row_in_a_sampling_batch_still_picks_argmax():
    # The tie at ids 1 and 2 must go to 1, as argmax does.
    logits = torch.tensor([[0.0, 3.0, 3.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

    for _ in range(50):
        assert sample(logits, [greedy(), SamplingParams()])[0] == 1


def test_top_k_never_leaves_the_k_most_likely_tokens():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])

    assert set(counts(logits, SamplingParams(top_k=2))) == {3, 4}


def test_top_p_keeps_the_fewest_tokens_that_reach_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))

    # 0.5 alone is below 0.7, 0.5 + 0.3 reaches it.
    assert set(counts(logits, SamplingParams(top_p=0.7))) == {0, 1}


def test_low_temperature_is_nearly_greedy():
    logits = torch.tensor([[0.0, 1.0, 2.0]])

    assert set(counts(logits, SamplingParams(temperature=1e-3), n=200)) == {2}


def test_high_temperature_flattens_the_distribution():
    # At temperature 1, id 0 has probability e^-5 < 1%.
    logits = torch.tensor([[0.0, 5.0]])

    assert counts(logits, SamplingParams(temperature=100.0))[0] > 0.4 * 2000


@pytest.mark.parametrize("seed", [None, 0])
def test_samples_follow_the_softmax_distribution(seed):
    probs = [0.6, 0.3, 0.1]
    n = 20_000

    freq = counts(torch.log(torch.tensor([probs])), SamplingParams(sampling_seed=seed), n=n)

    for token, p in enumerate(probs):
        assert abs(freq[token] / n - p) < 0.02


def test_seeded_row_does_not_depend_on_its_batch_mates():
    logits = torch.randn(4, 1000)
    seeded = SamplingParams(sampling_seed=1234)

    (alone,) = sample(logits[2:3], [seeded], positions=[7])
    batched = sample(logits, [SamplingParams(), greedy(), seeded, SamplingParams(top_k=5)], positions=[3, 9, 7, 1])

    assert batched[2] == alone


def test_apply_top_k_top_p_zeroes_dropped_tokens_in_vocab_order():
    probs = torch.tensor([[0.1, 0.4, 0.2, 0.3]] * 4)
    top_ks = torch.tensor([2, 1 << 30, 1 << 30, 1], dtype=torch.int32)
    top_ps = torch.tensor([1.0, 0.5, 0.3, 1.0])

    out = apply_top_k_top_p(probs, top_ks, top_ps)

    expected = torch.tensor(
        [
            [0.0, 0.4, 0.0, 0.3],  # top_k 2
            [0.0, 0.4, 0.0, 0.3],  # top_p 0.5: 0.4 is not enough, 0.4 + 0.3 is
            [0.0, 0.4, 0.0, 0.0],  # top_p 0.3: 0.4 alone reaches it
            [0.0, 0.4, 0.0, 0.0],  # top_k 1
        ]
    )
    torch.testing.assert_close(out, expected)


def test_apply_top_k_top_p_always_keeps_the_top_token():
    probs = torch.tensor([[0.2, 0.5, 0.3]])

    out = apply_top_k_top_p(probs, torch.tensor([1 << 30], dtype=torch.int32), torch.tensor([1e-6]))

    torch.testing.assert_close(out, torch.tensor([[0.0, 0.5, 0.0]]))


def test_same_seed_and_position_pick_the_same_token():
    logprobs = torch.log_softmax(torch.zeros(1, 1000), dim=-1)
    seed, position = torch.tensor([42]), torch.tensor([5])

    first = multinomial_with_seed(logprobs, seed, position)

    assert first.shape == (1, 1)
    assert torch.equal(first, multinomial_with_seed(logprobs, seed, position))


def test_same_seed_at_different_positions_picks_different_tokens():
    # Otherwise a seeded request would repeat one token forever.
    logprobs = torch.log_softmax(torch.zeros(20, 1000), dim=-1)

    tokens = multinomial_with_seed(logprobs, torch.full((20,), 42), torch.arange(20))

    assert len(set(tokens.view(-1).tolist())) > 1


def test_dropped_token_never_wins():
    logprobs = torch.tensor([[float("-inf"), 0.0, float("-inf")]]).repeat(200, 1)

    tokens = multinomial_with_seed(logprobs, torch.arange(200), torch.zeros(200, dtype=torch.long))

    assert tokens.view(-1).tolist() == [1] * 200
