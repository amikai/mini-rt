import json
import socket
import threading
import time
from itertools import pairwise

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from engine import Engine
from req import ReqStatus
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
    assert meta["finish_reason"]["type"] in {"stop", "length"}
    assert meta["prompt_tokens"] == len(engine.tokenizer.encode("Hello"))
    assert meta["completion_tokens"] == len(body["output_ids"])
    assert meta["id"]


def test_length_finish_reason_reports_limit(client, engine, monkeypatch):
    # Never emit EOS, so the request must hit max_new_tokens.
    monkeypatch.setattr(engine.scheduler, "run_batch", lambda batch: 0)

    body = client.post("/generate", json={"text": "Hello", "sampling_params": {"max_new_tokens": 3}}).json()

    assert body["output_ids"] == [0, 0, 0]
    assert body["meta_info"]["finish_reason"] == {"type": "length", "length": 3}


def test_eos_finish_reason_reports_matched_id(client, engine, monkeypatch):
    eos = next(iter(engine.model_runner.eos_token_ids))
    monkeypatch.setattr(engine.scheduler, "run_batch", lambda batch: eos)

    body = client.post("/generate", json={"text": "Hello"}).json()

    assert body["meta_info"]["finish_reason"] == {"type": "stop", "matched": eos}


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

    eos = next(iter(engine.model_runner.eos_token_ids))

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


def stream_events(client, body) -> list[str]:
    """POST with stream=true and return the data payload of every SSE message."""
    with client.stream("POST", "/generate", json={**body, "stream": True}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in resp.iter_lines() if line]
    assert all(line.startswith("data: ") for line in lines)
    return [line.removeprefix("data: ") for line in lines]


def test_stream_sends_one_chunk_per_token_then_done(client, monkeypatch, engine):
    monkeypatch.setattr(engine.scheduler, "run_batch", lambda batch: 0)

    events = stream_events(client, {"text": "Hello", "sampling_params": {"max_new_tokens": 3}})

    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert [c["output_ids"] for c in chunks] == [[0], [0, 0], [0, 0, 0]]
    assert [c["meta_info"]["finish_reason"] for c in chunks] == [None, None, {"type": "length", "length": 3}]
    assert [c["meta_info"]["completion_tokens"] for c in chunks] == [1, 2, 3]


def test_stream_matches_non_stream(client):
    body = {"text": "The capital of France is", "sampling_params": {"max_new_tokens": 8}}

    plain = client.post("/generate", json=body).json()
    chunks = [json.loads(e) for e in stream_events(client, body)[:-1]]

    last = chunks[-1]
    assert last["output_ids"] == plain["output_ids"]
    assert last["text"] == plain["text"]
    assert last["meta_info"]["finish_reason"] == plain["meta_info"]["finish_reason"]
    # text is everything so far, so each chunk extends the previous one.
    texts = [c["text"] for c in chunks]
    assert all(b.startswith(a) for a, b in pairwise(texts))


def test_stream_error_is_sent_as_a_chunk(client, engine, monkeypatch):
    def failing_run(batch):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(engine.scheduler, "run_batch", failing_run)

    events = stream_events(client, {"text": "Hello"})

    assert json.loads(events[0]) == {"error": {"message": "model crashed"}}
    assert events[-1] == "[DONE]"


def test_stream_invalid_max_new_tokens_is_client_error(client):
    resp = client.post("/generate", json={"text": "Hello", "stream": True, "sampling_params": {"max_new_tokens": 0}})

    assert 400 <= resp.status_code < 500


@pytest.fixture
def live_url(engine):
    # TestClient cannot drop a connection mid-stream, so run a real server.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(create_app(engine), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def endless(engine, monkeypatch):
    """Requests never finish on their own; returns the list of submitted Reqs."""

    def slow_run(batch):
        time.sleep(0.01)
        return 0  # never EOS

    monkeypatch.setattr(engine.scheduler, "run_batch", slow_run)
    reqs = []
    submit = engine.submit

    def recording_submit(*args, **kwargs):
        reqs.append(submit(*args, **kwargs))
        return reqs[-1]

    monkeypatch.setattr(engine, "submit", recording_submit)
    return reqs


def open_stream(url: str, prompt: str):
    body = {"text": prompt, "stream": True, "sampling_params": {"max_new_tokens": 100_000}}
    return httpx.stream("POST", f"{url}/generate", json=body, timeout=10)


def test_disconnect_aborts_running_request(live_url, endless):
    with open_stream(live_url, "A") as resp:
        next(resp.iter_lines())  # A is running
    # Leaving the block closes the connection.

    (a,) = endless
    assert a.done.wait(timeout=5)
    assert a.status is ReqStatus.ABORTED


def test_disconnect_aborts_waiting_request(live_url, endless):
    with open_stream(live_url, "A") as a_resp:
        # Keep the iterator: a discarded one is closed on garbage collection, which drops the connection.
        a_lines = a_resp.iter_lines()
        next(a_lines)  # A is running
        with open_stream(live_url, "B") as b_resp:
            assert b_resp.status_code == 200  # headers sent, B queued behind A
        a, b = endless
        assert b.done.wait(timeout=5)
        assert b.status is ReqStatus.ABORTED
        assert b.output_ids == []
        assert a.status is ReqStatus.RUNNING

    assert a.done.wait(timeout=5)
    assert a.status is ReqStatus.ABORTED
