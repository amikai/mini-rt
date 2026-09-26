"""M7: same HTTP server as M6, plus `stream: true` for Server-Sent Events and abort on client disconnect.

One process: server, tokenizer, scheduler thread, and model live together.
"""

import argparse
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from engine import Engine
from req import BaseFinishReason, Req


class SamplingParams(BaseModel):
    """How to generate. Only max_new_tokens for now; temperature etc. come later."""

    max_new_tokens: int = 32  # upper bound on generated tokens; EOS may stop earlier


class GenerateReqInput(BaseModel):
    """POST /generate request body. Subset of SGLang's GenerateReqInput (managers/io_struct.py)."""

    text: str  # raw prompt, tokenized as-is (no chat template)
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
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
    """Yields one chunk per step from req.out_queue, then [DONE]."""
    # TODO(M7-5): loop over req.out_queue items and turn each into SSE messages.
    # - Read with `await asyncio.to_thread(req.out_queue.get)`: the blocking get runs in a worker thread,
    #   so a client disconnect can cancel the await.
    # - Keep your own output_ids list and one IncrementalDetokenizer; req.output_ids may be ahead of what you read.
    # - FINISH_ERROR: headers are already sent as 200, so yield sse({"error": {"message": ...}}), then [DONE].
    # - Else: add the token (if any), decode, yield sse(make_response(...).model_dump_json()).
    # - The request ended: yield [DONE] and stop.
    raise NotImplementedError
    yield  # makes this an async generator until the TODO is done


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI()

    # Plain `def`: FastAPI runs it in a threadpool, so waiting for the scheduler does not freeze the event loop.
    @app.post("/generate", response_model=None)
    def generate(obj: GenerateReqInput) -> GenerateResponse | StreamingResponse:
        try:
            if obj.stream:
                req = engine.submit(obj.text, obj.sampling_params.max_new_tokens)
            else:
                req = engine.generate(obj.text, obj.sampling_params.max_new_tokens)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if obj.stream:
            # The background task runs after the stream ends, also on disconnect; aborting a finished request is a no-op.
            return StreamingResponse(
                stream_results(engine, req),
                media_type="text/event-stream",
                background=BackgroundTask(engine.abort, req),
            )
        return make_response(req, engine.decode(req), req.output_ids, req.finished_reason)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)  # SGLang's default port
    args = parser.parse_args()

    uvicorn.run(create_app(Engine(model_id=args.model)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
