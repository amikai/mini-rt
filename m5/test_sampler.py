import torch

from sampler import Sampler


def test_greedy_picks_argmax():
    logits = torch.tensor([0.1, 2.0, -1.0, 1.9])

    assert Sampler()(logits) == 1


def test_returns_python_int():
    assert type(Sampler()(torch.randn(10))) is int
