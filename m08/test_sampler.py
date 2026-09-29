import torch

from sampler import Sampler


def test_greedy_picks_argmax_per_row():
    logits = torch.tensor([[0.1, 2.0, -1.0, 1.9], [3.0, 0.0, 0.0, 0.0]])

    assert Sampler()(logits) == [1, 0]


def test_returns_python_ints():
    assert all(type(t) is int for t in Sampler()(torch.randn(3, 10)))
