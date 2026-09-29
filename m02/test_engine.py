import threading

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from engine import Engine, EngineBusyError


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    # Tiny random Qwen3 with the real tokenizer: same code path as Qwen3-0.6B, fast on CPU.
    model_dir = tmp_path_factory.mktemp("tiny-qwen3")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(
        vocab_size=len(tokenizer),  # keep full vocab so every token ID fits the embedding
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        eos_token_id=tokenizer.eos_token_id,  # random config has no EOS; Engine needs one
        pad_token_id=tokenizer.pad_token_id,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return Engine(model_id=str(model_dir), device="cpu")


def test_generate_fills_request(engine):
    req = engine.generate("The capital of France is", max_new_tokens=5)

    assert req.prompt == "The capital of France is"
    assert req.input_ids == engine.tokenizer.encode(req.prompt)
    assert 1 <= len(req.output_ids) <= 5
    assert req.finish_reason in {"stop", "length"}
    assert isinstance(engine.decode(req), str)


def test_finish_reason_matches_last_token(engine):
    req = engine.generate("Hello", max_new_tokens=3)

    if req.output_ids[-1] in engine.eos_token_ids:
        assert req.finish_reason == "stop"
    else:
        assert req.finish_reason == "length"
        assert len(req.output_ids) == 3


def test_each_request_gets_unique_rid(engine):
    a = engine.generate("Hello", max_new_tokens=1)
    b = engine.generate("Hello", max_new_tokens=1)

    assert a.rid != b.rid


def test_engine_is_idle_after_request(engine):
    engine.generate("Hello", max_new_tokens=1)

    assert engine.running_req is None


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_rejects_invalid_max_new_tokens(engine, max_new_tokens):
    with pytest.raises(ValueError):
        engine.generate("Hello", max_new_tokens=max_new_tokens)

    assert engine.running_req is None


def test_rejects_request_while_busy(engine, monkeypatch):
    # Block A inside _run until the test releases it, so B arrives while A is running.
    admitted = threading.Event()
    release = threading.Event()

    def blocking_run(req):
        admitted.set()
        release.wait(timeout=10)
        req.output_ids = [0]
        req.finish_reason = "length"

    monkeypatch.setattr(engine, "_run", blocking_run)

    results = {}
    a = threading.Thread(target=lambda: results.setdefault("A", engine.generate("A", max_new_tokens=1)))
    a.start()
    assert admitted.wait(timeout=10)
    assert engine.running_req is not None

    with pytest.raises(EngineBusyError):
        engine.generate("B", max_new_tokens=1)

    release.set()
    a.join(timeout=10)
    assert results["A"].finish_reason == "length"
    assert engine.running_req is None


def test_engine_recovers_after_run_error(engine, monkeypatch):
    def failing_run(req):
        raise RuntimeError("model failed")

    monkeypatch.setattr(engine, "_run", failing_run)
    with pytest.raises(RuntimeError):
        engine.generate("Hello", max_new_tokens=1)
    monkeypatch.undo()

    # The lock must be released, so the next request is admitted.
    assert engine.running_req is None
    assert engine.generate("Hello", max_new_tokens=1).finish_reason in {"stop", "length"}
