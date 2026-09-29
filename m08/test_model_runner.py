import pytest
import torch

from model_runner import ModelRunner
from req import Req


@pytest.fixture(scope="module")
def runner(tiny_model_dir):
    return ModelRunner(tiny_model_dir, device="cpu")


def make_req(input_ids: list[int], output_ids: list[int] | None = None) -> Req:
    return Req(rid="r", prompt="", input_ids=input_ids, max_new_tokens=4, output_ids=output_ids or [])


def test_prepare_inputs_left_pads_to_longest(runner):
    batch = [make_req([11, 12, 13], output_ids=[14]), make_req([21])]

    input_ids, attention_mask, position_ids = runner.prepare_inputs(batch)

    assert input_ids.shape == attention_mask.shape == position_ids.shape == (2, 4)
    assert input_ids.dtype == attention_mask.dtype == position_ids.dtype == torch.long
    # Real tokens are right-aligned; only the mask says where padding is.
    assert attention_mask.tolist() == [[1, 1, 1, 1], [0, 0, 0, 1]]
    assert input_ids[0].tolist() == [11, 12, 13, 14]
    assert input_ids[1, -1].item() == 21
    # Positions count from each row's first real token, not from column 0.
    assert position_ids[0].tolist() == [0, 1, 2, 3]
    assert position_ids[1, -1].item() == 0


def test_forward_returns_one_row_per_request(runner):
    batch = [make_req([1, 2, 3]), make_req([4])]

    logits = runner.forward(batch)

    assert logits.shape == (2, runner.model.config.vocab_size)


def test_forward_includes_output_ids(runner):
    # Generated tokens are part of the next step's input.
    a = make_req([1, 2, 3])
    b = make_req([1, 2], output_ids=[3])

    assert torch.equal(runner.forward([a]), runner.forward([b]))


def test_batched_logits_match_each_request_alone(runner):
    # Padding must not change any request's result.
    batch = [make_req([9707, 11, 1879]), make_req([1, 2, 3, 4, 5, 6, 7], output_ids=[8]), make_req([42])]

    batched = runner.forward(batch)

    for row, req in zip(batched, batch, strict=True):
        # bf16 kernels may round a little differently for a different batch shape.
        torch.testing.assert_close(row, runner.forward([req])[0], atol=1e-2, rtol=0)


def test_forward_tells_the_model_where_padding_and_positions_are(runner, monkeypatch):
    # RoPE only sees distances, so logits barely change without position_ids; check what forward hands the model.
    seen = {}
    model_forward = runner.model.forward

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return model_forward(*args, **kwargs)

    monkeypatch.setattr(runner.model, "forward", spy)
    batch = [make_req([1, 2, 3]), make_req([4])]
    runner.forward(batch)

    _, attention_mask, position_ids = runner.prepare_inputs(batch)
    assert torch.equal(seen["attention_mask"], attention_mask)
    assert torch.equal(seen["position_ids"], position_ids)


def test_sample_returns_one_token_per_request(runner):
    batch = [make_req([1, 2, 3]), make_req([4])]

    tokens = runner.sample(runner.forward(batch), batch)

    assert len(tokens) == 2
    assert all(type(t) is int for t in tokens)


def test_step_loop_matches_hf_greedy_generate(runner):
    # forward + sample, repeated, must equal HF greedy decoding.
    input_ids = [9707, 11, 1879]
    req = make_req(input_ids)
    for _ in range(8):
        (token,) = runner.sample(runner.forward([req]), [req])
        req.output_ids.append(token)

    expected = runner.model.generate(
        torch.tensor([input_ids]), max_new_tokens=8, do_sample=False, use_cache=False, eos_token_id=None
    )[0, len(input_ids) :].tolist()
    assert req.output_ids == expected
