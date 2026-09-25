"""M5: same HTTP server as M4. Only the Engine internals changed.

One process: server, tokenizer, scheduler thread, and model live together.
"""

import argparse

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine import Engine


class SamplingParams(BaseModel):
    """How to generate. Only max_new_tokens for now; temperature etc. arrive in M9."""

    max_new_tokens: int = 32  # upper bound on generated tokens; EOS may stop earlier


class GenerateReqInput(BaseModel):
    """POST /generate request body. Subset of SGLang's GenerateReqInput (managers/io_struct.py)."""

    text: str  # raw prompt, tokenized as-is (no chat template)
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)


class MetaInfo(BaseModel):
    """Bookkeeping about the request, not the generated content."""

    id: str  # request id, Req.rid
    finish_reason: str  # "stop" (hit EOS) or "length" (hit max_new_tokens)
    prompt_tokens: int  # len(Req.input_ids)
    completion_tokens: int  # len(Req.output_ids)


class GenerateResponse(BaseModel):
    """POST /generate response body."""

    text: str  # decoded output_ids only, prompt not included, special tokens skipped
    output_ids: list[int]  # generated token IDs only, prompt excluded; may end with EOS
    meta_info: MetaInfo


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI()

    # Plain `def`: FastAPI runs it in a threadpool, so waiting for the scheduler does not freeze the event loop.
    @app.post("/generate")
    def generate(obj: GenerateReqInput) -> GenerateResponse:
        try:
            req = engine.generate(obj.text, obj.sampling_params.max_new_tokens)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return GenerateResponse(
            text=engine.decode(req),
            output_ids=req.output_ids,
            meta_info=MetaInfo(
                id=req.rid,
                finish_reason=req.finish_reason,
                prompt_tokens=len(req.input_ids),
                completion_tokens=len(req.output_ids),
            ),
        )

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
