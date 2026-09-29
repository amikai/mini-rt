import threading
import time

import pytest
from fastapi.testclient import TestClient

from engine import Engine
from server import create_app


@pytest.fixture(scope="session")
def engine(tiny_model_dir):
    return Engine(model_id=tiny_model_dir, device="cpu")


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


def test_concurrent_requests_queue_instead_of_503(client, engine, monkeypatch):
    # Hold A inside run_batch so B and C arrive while A is running.
    admitted = threading.Event()
    release = threading.Event()
    order = []

    eos = next(iter(engine.scheduler.eos_token_ids))

    def blocking_run(batch):
        if batch.prompt == "A":
            admitted.set()
            release.wait(timeout=10)
        order.append(batch.prompt)
        return eos  # finish in one step

    monkeypatch.setattr(engine.scheduler, "run_batch", blocking_run)

    results = {}

    def post(prompt):
        results[prompt] = client.post("/generate", json={"text": prompt})

    a = threading.Thread(target=post, args=("A",))
    a.start()
    assert admitted.wait(timeout=10)

    # B before C: wait until B is queued so the submit order is fixed.
    b = threading.Thread(target=post, args=("B",))
    b.start()
    while engine.scheduler.recv_queue.qsize() + len(engine.scheduler.waiting_queue) < 1:
        time.sleep(0.01)
    c = threading.Thread(target=post, args=("C",))
    c.start()
    while engine.scheduler.recv_queue.qsize() + len(engine.scheduler.waiting_queue) < 2:
        time.sleep(0.01)

    release.set()
    for t in (a, b, c):
        t.join(timeout=10)

    assert {p: r.status_code for p, r in results.items()} == {"A": 200, "B": 200, "C": 200}
    assert order == ["A", "B", "C"]


def test_run_error_returns_500_and_server_keeps_serving(engine, monkeypatch):
    # raise_server_exceptions=False: get the 500 response instead of the exception.
    client = TestClient(create_app(engine), raise_server_exceptions=False)

    def failing_run(batch):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(engine.scheduler, "run_batch", failing_run)
    assert client.post("/generate", json={"text": "Hello"}).status_code == 500

    monkeypatch.undo()
    assert client.post("/generate", json={"text": "Hello"}).status_code == 200
