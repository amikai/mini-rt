import torch

from req import Req
from sampling_batch_info import SamplingBatchInfo
from sampling_params import TOP_K_ALL, SamplingParams


def make_req(**kwargs) -> Req:
    return Req(rid="r", prompt="", input_ids=[1], sampling_params=SamplingParams(**kwargs))


def test_one_row_per_request_in_batch_order():
    reqs = [make_req(temperature=0.5, top_p=0.9, top_k=10), make_req()]

    info = SamplingBatchInfo.from_reqs(reqs, "cpu")

    assert info.temperatures.shape == (2, 1)
    assert info.temperatures.dtype == info.top_ps.dtype == torch.float32
    assert info.top_ks.dtype == torch.int32
    assert info.temperatures.view(-1).tolist() == [0.5, 1.0]
    torch.testing.assert_close(info.top_ps, torch.tensor([0.9, 1.0]))
    assert info.top_ks.tolist() == [10, TOP_K_ALL]


def test_greedy_request_is_a_top_k_one_row():
    info = SamplingBatchInfo.from_reqs([make_req(temperature=0.0)], "cpu")

    assert info.top_ks.tolist() == [1]
    assert info.temperatures.view(-1).tolist() == [1.0]


def test_all_greedy_only_when_every_row_is_greedy():
    greedy = [make_req(temperature=0.0), make_req(top_k=1)]

    assert SamplingBatchInfo.from_reqs(greedy, "cpu").is_all_greedy
    assert not SamplingBatchInfo.from_reqs([*greedy, make_req()], "cpu").is_all_greedy


def test_no_seed_in_the_batch_means_no_seed_tensor():
    info = SamplingBatchInfo.from_reqs([make_req(), make_req()], "cpu")

    assert info.sampling_seed is None


def test_seeded_rows_keep_their_seed_and_unseeded_rows_get_a_fresh_one():
    reqs = [make_req(sampling_seed=7), make_req(), make_req()]

    first = SamplingBatchInfo.from_reqs(reqs, "cpu").sampling_seed
    second = SamplingBatchInfo.from_reqs(reqs, "cpu").sampling_seed

    assert first.shape == (3,)
    assert first.dtype == torch.int64
    assert first[0].item() == second[0].item() == 7
    # Unseeded rows stay random from step to step.
    assert first[1:].tolist() != second[1:].tolist()
