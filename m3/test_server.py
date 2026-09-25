import threading

import pytest
from fastapi.testclient import TestClient
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from engine import Engine
from server import create_app


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    # Tiny random Qwen3 with the real tokenizer, same as m2 tests.
    model_dir = tmp_path_factory.mktemp("tiny-qwen3")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return Engine(model_id=str(model_dir), device="cpu")


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def test_generate_returns_sglang_format(client, engine):
    resp = client.post("/generate", json={"text": "Hello", "sampling_params": {"max_new_tokens": 5}})

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["text"], str)
    assert 1 <= len(body["output_ids"]) <= 5
    meta = body["meta_info"]
    assert meta["finish_reason"] in {"stop", "length"}
    assert meta["prompt_tokens"] == len(engine.tokenizer.encode("Hello"))
    assert meta["completion_tokens"] == len(body["output_ids"])
    assert meta["id"]


def test_sampling_params_is_optional(client):
    resp = client.post("/generate", json={"text": "Hello"})

    assert resp.status_code == 200


def test_missing_text_is_rejected(client):
    resp = client.post("/generate", json={"sampling_params": {"max_new_tokens": 5}})

    assert resp.status_code == 422


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_invalid_max_new_tokens_is_client_error(client, max_new_tokens):
    resp = client.post("/generate", json={"text": "Hello", "sampling_params": {"max_new_tokens": max_new_tokens}})

    # 400 or 422 is your call; it must not be 500.
    assert 400 <= resp.status_code < 500


def test_busy_returns_503(client, engine, monkeypatch):
    # Block A inside _run so B arrives while A is running.
    admitted = threading.Event()
    release = threading.Event()

    def blocking_run(req):
        admitted.set()
        release.wait(timeout=10)
        req.output_ids = [0]
        req.finish_reason = "length"

    monkeypatch.setattr(engine, "_run", blocking_run)

    results = {}
    a = threading.Thread(target=lambda: results.setdefault("A", client.post("/generate", json={"text": "A"})))
    a.start()
    assert admitted.wait(timeout=10)

    b = client.post("/generate", json={"text": "B"})
    assert b.status_code == 503

    release.set()
    a.join(timeout=10)
    assert results["A"].status_code == 200
