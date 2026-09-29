import pytest
import torch

from model_runner import ModelRunner
from req import Req


@pytest.fixture(scope="module")
def runner(tiny_model_dir):
    return ModelRunner(tiny_model_dir, device="cpu")


def test_forward_returns_last_position_logits(runner):
    req = Req(rid="r", prompt="", input_ids=[1, 2, 3], max_new_tokens=4)

    logits = runner.forward(req)

    assert logits.shape == (runner.model.config.vocab_size,)


def test_forward_includes_output_ids(runner):
    # Generated tokens are part of the next step's input.
    a = Req(rid="a", prompt="", input_ids=[1, 2, 3], max_new_tokens=4)
    b = Req(rid="b", prompt="", input_ids=[1, 2], max_new_tokens=4, output_ids=[3])

    assert torch.equal(runner.forward(a), runner.forward(b))


def test_step_loop_matches_hf_greedy_generate(runner):
    # forward + sample, repeated, must equal HF greedy decoding.
    input_ids = [9707, 11, 1879]
    req = Req(rid="r", prompt="", input_ids=input_ids, max_new_tokens=8)
    for _ in range(req.max_new_tokens):
        req.output_ids.append(runner.sample(runner.forward(req), req))

    expected = runner.model.generate(
        torch.tensor([input_ids]), max_new_tokens=8, do_sample=False, use_cache=False, eos_token_id=None
    )[0, len(input_ids) :].tolist()
    assert req.output_ids == expected
