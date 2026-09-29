"""M10: same HTTP server as M9. sampling_params now takes temperature, top_p, top_k, and sampling_seed.

One process: server, tokenizer, scheduler thread, and model live together.
"""

import argparse
import asyncio
import json
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from detokenizer import IncrementalDetokenizer
from engine import Engine
from req import FINISH_ERROR, BaseFinishReason, Req
from sampling_params import SamplingParams


class GenerateReqInput(BaseModel):
    """POST /generate request body. Subset of SGLang's GenerateReqInput (managers/io_struct.py)."""

    text: str  # raw prompt, tokenized as-is (no chat template)
    # Keyword arguments of SamplingParams, as a plain dict like SGLang's.
    sampling_params: dict = Field(default_factory=dict)
    stream: bool = False  # True: one SSE chunk per step instead of one JSON body at the end


class MetaInfo(BaseModel):
    """Bookkeeping about the request, not the generated content."""

    id: str  # request id, Req.rid
    # {"type": "stop", ...}, {"type": "length", ...}, {"type": "abort", ...}; None on stream chunks before the last
    finish_reason: dict | None
    prompt_tokens: int  # len(Req.input_ids)
    completion_tokens: int  # len(output_ids)


class GenerateResponse(BaseModel):
    """POST /generate response body, and the body of each stream chunk."""

    text: str  # decoded output_ids only, prompt not included, special tokens skipped
    output_ids: list[int]  # generated token IDs only, prompt excluded; may end with EOS
    meta_info: MetaInfo


def make_response(
    req: Req, text: str, output_ids: list[int], finish_reason: BaseFinishReason | None
) -> GenerateResponse:
    return GenerateResponse(
        text=text,
        output_ids=output_ids,
        meta_info=MetaInfo(
            id=req.rid,
            finish_reason=finish_reason.to_json() if finish_reason else None,
            prompt_tokens=len(req.input_ids),
            completion_tokens=len(output_ids),
        ),
    )


def sse(data: str) -> str:
    """One Server-Sent Events message."""
    return f"data: {data}\n\n"


async def stream_results(engine: Engine, req: Req) -> AsyncIterator[str]:
    """Yields one chunk per step from req.out_queue, then [DONE].

    SSE defines no end-of-stream message, so the end is `data: [DONE]`, as in OpenAI's API and SGLang.
    """
    detokenizer = IncrementalDetokenizer(engine.tokenizer)
    # Our own copy; req.output_ids may already be ahead of what we have read.
    output_ids: list[int] = []
    while True:
        # Blocking get in a worker thread, so a client disconnect can cancel this await.
        token_id, finish_reason = await asyncio.to_thread(req.out_queue.get)
        if isinstance(finish_reason, FINISH_ERROR):
            # Headers are already sent as 200, so the error travels as a chunk, as in SGLang.
            yield sse(json.dumps({"error": {"message": str(finish_reason.error)}}))
            yield sse("[DONE]")
            return
        if token_id is not None:
            output_ids.append(token_id)
        text = detokenizer.decode(output_ids, finished=finish_reason is not None)
        yield sse(make_response(req, text, output_ids, finish_reason).model_dump_json())
        if finish_reason is not None:
            yield sse("[DONE]")
            return


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI()

    # Plain `def`: FastAPI runs it in a threadpool, so waiting for the scheduler does not freeze the event loop.
    @app.post("/generate", response_model=None)
    def generate(obj: GenerateReqInput) -> GenerateResponse | StreamingResponse:
        # Validate before submit, so a model error raised by generate() cannot turn into a 400.
        try:
            # TypeError: unknown key or wrong type. ValueError: out-of-range value.
            sampling_params = SamplingParams(**obj.sampling_params)
            sampling_params.verify()
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if obj.stream:
            req = engine.submit(obj.text, sampling_params)
            # Starlette runs the stream next to a listener for http.disconnect; whichever ends first cancels the other.
            return StreamingResponse(
                stream_results(engine, req),
                media_type="text/event-stream",
                # Runs after either ending. Aborting a request that already finished is a no-op.
                background=BackgroundTask(engine.abort, req),
            )
        req = engine.generate(obj.text, sampling_params)
        return make_response(req, engine.decode(req), req.output_ids, req.finished_reason)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)  # SGLang's default port
    parser.add_argument("--max-batch-size", type=int, default=8)
    args = parser.parse_args()

    engine = Engine(model_id=args.model, max_batch_size=args.max_batch_size)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
