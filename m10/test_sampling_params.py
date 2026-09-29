import pytest

from sampling_params import TOP_K_ALL, SamplingParams


def test_default_samples_from_the_whole_vocabulary():
    params = SamplingParams()

    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k == TOP_K_ALL
    assert params.sampling_seed is None


def test_temperature_zero_becomes_top_k_one():
    params = SamplingParams(temperature=0.0, top_k=50)

    assert params.top_k == 1
    # Logits are divided by temperature, so 0 must not reach the Sampler.
    assert params.temperature == 1.0


def test_top_k_minus_one_keeps_every_token():
    assert SamplingParams(top_k=-1).top_k == TOP_K_ALL


def test_positive_temperature_and_top_k_are_kept():
    params = SamplingParams(temperature=0.7, top_k=5)

    assert params.temperature == 0.7
    assert params.top_k == 5


def test_verify_accepts_valid_params():
    SamplingParams(temperature=0.0, top_p=0.9, top_k=1, sampling_seed=1).verify()
    SamplingParams(temperature=2.0, top_p=1.0, top_k=-1).verify()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"temperature": -0.1},
        {"temperature": float("inf")},
        {"temperature": float("nan")},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_k": 0},
        {"top_k": -2},
    ],
)
def test_verify_rejects_values_the_sampler_cannot_use(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs).verify()
