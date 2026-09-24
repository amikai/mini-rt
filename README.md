# mini-rt

A minimal LLM inference runtime, built from scratch one concept at a time.

## Why This Repo Exists

I built this repo to learn what [SGLang](https://github.com/sgl-project/sglang) does and why it is designed the way it is.

Reading a production inference engine directly is hard. Abstractions such as the scheduler, `ScheduleBatch`, `ForwardBatch`, the KV cache pool, and the radix cache all exist to solve specific problems, and the code rarely tells you what those problems were.

So this repo does not clone SGLang feature by feature. It starts from a bare Qwen3 generation loop and runs into the same problems SGLang was built to solve, one at a time, solving each one by hand. Each milestone adds exactly one core idea and answers one new systems question.

After about M14, the SGLang scheduler, model runner, and KV cache code should read as familiar solutions to problems already met here.

## Model

The whole roadmap uses [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). It is small enough to run on a laptop, and its architecture matches the models SGLang serves in production: RoPE, grouped-query attention (GQA), RMSNorm, and a SwiGLU MLP. Using the same model from M1 onward means the tokenizer, EOS tokens, KV cache shape, and position handling never change underneath the runtime while it is being built. M18 then adds GPT-2 as a second model to test that the runtime does not depend on Qwen3.

## Roadmap at a Glance

| Milestone | Question | Answer |
|-----------|----------|--------|
| M1 | How does a model generate tokens? | Autoregressive loop |
| M2 | How do I turn it into a service? | Request + Engine |
| M3 | What if many requests arrive at once? | Scheduler |
| M4 | Who is responsible for what? | Layer separation |
| M5 | What is a request over time? | State machine |
| M6 | Running one request at a time wastes the GPU | Static batching |
| M7 | Some requests in a batch finish early | Continuous batching |
| M8 | Each request wants different sampling settings | Per-request sampling |
| M9 | Where is the time actually going? | Profiling |
| M10 | What does the model actually compute? | Own model implementation |
| M11 | Why recompute everything every step? | KV cache |
| M12 | Why is the first step different from the rest? | Prefill / decode |
| M13 | Should scheduling data and execution data be the same? | ScheduleBatch / ForwardBatch |
| M14 | Who manages the growing KV memory? | KV cache manager |
| M15 | Does the scheduler need to know about the GPU? | Backend boundary |
| M16 | What happens below the operator? | Custom GPU kernel |
| M17 | Contiguous memory is hard to manage | Paged KV cache + paged attention |
| M18 | Is the runtime really model-agnostic? | Second model: GPT-2 |
| M19 | Why can't requests share computation? | Prefix cache |
| M20 | How do I manage many prefixes? | Radix cache + schedule policy |
| M21 | A huge prompt is blocking the GPU | Chunked prefill |
| M22 | The CPU and GPU keep waiting on each other | Overlap scheduling |
| M23 | — | Mini inference runtime |

---

## M1 — Minimal Qwen3

**Build**

Control Qwen3 inference directly instead of calling `model.generate()`.

```text
text → tokenizer → Qwen3 → logits → argmax → next token
```

- Python + PyTorch, Qwen3-0.6B in bf16
- Hugging Face tokenizer and pretrained model
- Plain-text prompts; no chat template or thinking mode yet
- Hand-written autoregressive generation loop with greedy decoding
- CPU / MPS / CUDA device selection
- Recompute the full sequence for every new token

```python
for _ in range(max_new_tokens):
    logits = model(input_ids).logits
    next_token = logits[:, -1].argmax(dim=-1)
    input_ids = torch.cat(...)
```

Out of scope: KV cache, batching, scheduling, HTTP, sampling, profiling, `model.generate()`.

**Learn**

- `model.forward()` is not the same as text generation.
- What `input_ids` and logits are, and how next-token prediction works.
- How autoregressive generation is built on top of a single forward pass.

---

## M2 — Single Request Runtime

**Build**

Turn `generate(prompt)` into `Request → Runtime → Model`.

- A `Request` with `id`, `prompt`, `input_ids`, `output_ids`, `max_new_tokens`
- An `Engine` that runs one request at a time
- If a request is already running, new requests are rejected as busy

Out of scope: queues, scheduling, batching, caching.

**Learn**

- Serving a model is different from calling a model.
- Request lifecycle, runtime state, capacity, and admission.

---

## M3 — FIFO Scheduler

**Build**

Queue requests instead of rejecting them.

- A `Scheduler` with a FIFO waiting queue
- One running request at a time; the next waiting request starts when it finishes
- No model or PyTorch logic inside the scheduler

```text
submit A, B, C  →  execution order: A, B, C
```

Out of scope: batching, KV cache, advanced scheduling policies.

**Learn**

- The scheduler decides *who runs*; the model decides *how to compute*.
- The first split between control plane and compute.

---

## M4 — Runtime Layer Separation

**Build**

No new features. Refactor into four components:

```text
Engine
  ├─ Scheduler     decides who runs
  ├─ ModelRunner   owns device placement and forward execution (tensor → logits)
  └─ Sampler       turns logits into the next token (greedy)
```

- No model logic in `Scheduler`, no queue logic in `ModelRunner`, no model knowledge in `Sampler`
- Behavior stays identical to M3

**Learn**

- Runtime orchestration, model execution, sampling, and scheduling are four separate responsibilities.
- This is the basic shape of SGLang and vLLM.

---

## M5 — Request State Machine

**Build**

Make a request a state machine instead of a plain data object.

```text
WAITING → RUNNING → FINISHED
                  ↘ FAILED
```

- Explicit state transitions in `Scheduler` and `Engine`
- Termination by EOS token, `max_new_tokens`, or exception
- Support more than one EOS token ID (Qwen3 uses both `<|im_end|>` and `<|endoftext|>`)

Out of scope: batching, KV cache.

**Learn**

- An inference request is a long-lived state machine, not a function call.
- This is the foundation for continuous batching, cancellation, streaming, and timeouts.

---

## M6 — Static Batching

**Build**

Serve many requests with one model forward pass.

- Scheduler picks up to `max_batch_size` waiting requests
- Padded batch tensor with attention masks for different sequence lengths
- A batch stays together until every request in it finishes; empty slots are not refilled

Out of scope: continuous batching, KV cache.

**Learn**

- Why GPU throughput depends on batching.
- How to handle mismatched sequence lengths with padding, masks, and a batch dimension.

---

## M7 — Continuous Batching

**Build**

Refill a batch slot as soon as a request finishes.

```text
A B C  →  _ B C  →  D B C
```

- Separate `waiting` and `running` sets
- Every generation step re-decides what runs
- Finished requests release their slot immediately
- Padded tensors are rebuilt every step, and the full sequence is still recomputed

Out of scope: KV cache.

**Learn**

- Continuous batching is a scheduling problem, not a Transformer problem.
- The scheduler's real job is to rebuild the workload on every step.

---

## M8 — Per-Request Sampling

**Build**

Replace greedy-only decoding with sampling parameters that belong to each request.

- `SamplingParams` on each `Request`: `temperature`, `top_k`, `top_p`, and an optional seed
- `temperature = 0` means greedy, so M1–M7 behavior is still reachable
- The `Sampler` handles one batch where every row can have different parameters
- Parameters are turned into per-row tensors, so sampling runs as batched tensor ops rather than a Python loop over requests

Out of scope: repetition penalties, logit bias, constrained or grammar-based decoding.

**Learn**

- Sampling settings are request state, not model state.
- Continuous batching mixes requests with different settings in the same batch, so the sampler must be batch-aware.
- How temperature, top-k, and top-p reshape the probability distribution.

---

## M9 — Profiling & Observability

**Build**

Measure everything. Optimize nothing.

Request-level metrics:

| Metric | Definition |
|--------|------------|
| E2E latency | finish − arrival |
| Queue latency | start of execution − arrival |
| TTFT | first token − arrival |
| ITL | token N time − token N−1 time |
| Output tokens/sec | generated tokens / generation time |

Stage-level timings: tokenization, queue wait, batch preparation, model forward, sampling, request update, detokenization.

Runtime state: waiting requests, running requests, batch size, sequence length, prompt tokens, generated tokens.

Compute amplification, which exposes the cost of having no cache:

```text
compute amplification = processed_tokens / generated_tokens
```

Optional `torch.profiler` integration for operator-level profiling.

**Learn**

- The core engineering loop: measure → understand → optimize → measure again.
- CPU wall-clock latency is not GPU execution time; asynchronous GPU execution must be synchronized before timing.

---

## M10 — Own Model Implementation

**Build**

Replace the Hugging Face model with a hand-written Qwen3. Load only the pretrained weights from Hugging Face.

- Token embedding and tied LM head
- RMSNorm
- Rotary position embedding (RoPE)
- Grouped-query attention (GQA): 16 query heads share 8 KV heads
- Per-head RMSNorm on Q and K, as Qwen3 does
- SwiGLU MLP
- A `ModelConfig` with `num_layers`, `num_heads`, `num_kv_heads`, `head_dim`, `vocab_size`, `max_context_len`, `eos_token_ids`, and `dtype`
- A model interface, `forward(batch) -> logits`, that `ModelRunner` calls without knowing the model type
- A model registry that selects the model implementation and its `ModelConfig` by name
- Attention as a shared module that takes Q, K, V and returns the attention output; RoPE and Q/K norm stay in the Qwen3 code
- Validate logits against the Hugging Face model within a tolerance
- Re-run the M9 benchmarks as the new baseline

Out of scope: KV cache, custom kernels, a second model architecture (see M18).

**Learn**

- A model is a fixed sequence of matrix operations with known shapes and costs.
- Attention is the only operation that mixes tokens; every other layer works on each token independently.
- How GQA sets the size of the KV cache, and how RoPE depends on correct token positions.
- The runtime can only change how attention reads K/V if it owns the model code.
- Which parts are model-specific (embedding, norm, MLP, position encoding) and which parts the runtime shares across models (attention, KV access).

Reference: [tiny-llm](https://skyzh.github.io/tiny-llm/) Week 1 builds the same Qwen3 pieces step by step.

---

## M11 — KV Cache

**Build**

Remove the redundant computation that M9 exposed.

- Per-request KV cache in contiguous tensors
- Each new token appends its K/V to the existing cache
- Continuous batching still works
- Output is validated against the non-cached implementation
- Re-run the M9 benchmarks and compare forward latency, ITL, processed tokens, and compute amplification against M10

Out of scope: paging, block allocation, prefix sharing, memory pooling.

**Learn**

- Why the KV cache exists: the causal mask means K/V for past tokens never change.
- Model execution now carries persistent state across steps.

---

## M12 — Prefill / Decode Separation

**Build**

- `ModelRunner.prefill()` and `ModelRunner.decode()` as separate paths
- New requests run prefill first, then decode on later steps
- Scheduler tracks whether each request needs prefill or decode

Out of scope: chunked prefill, prefill/decode disaggregation.

**Learn**

- Prefill builds the context and KV cache; decode reuses it to append one token.
- The two workloads have very different GPU characteristics: prefill is compute-bound, decode is memory-bound.

---

## M13 — ScheduleBatch / ForwardBatch

**Build**

Split the CPU-side scheduling representation from the device-side execution representation.

- `ScheduleBatch`: Python `Request` objects and scheduling metadata
- `ForwardBatch`: input tensors, positions, sequence lengths, cache metadata
- A clear conversion boundary before `ModelRunner` runs

```text
Scheduler → ScheduleBatch
──────── boundary ────────
ModelRunner → ForwardBatch → GPU
```

No new optimization behavior.

**Learn**

- Why control-plane data structures and execution data structures should be separate.
- This maps directly onto SGLang's internals.

---

## M14 — KV Cache Manager

**Build**

Move KV ownership from the request to the runtime.

- A central `KVCacheManager` with `allocate(req)`, `get(req)`, `free(req)`
- The manager sizes storage from `ModelConfig` (`num_layers`, `num_kv_heads`, `head_dim`, `dtype`), not from the model type
- Requests hold only a logical cache handle
- Allocation is still contiguous

Out of scope: paging, shared prefix caching.

**Learn**

- Request lifecycle and memory lifecycle are different things.
- This is the foundation for paged KV, radix cache, and OOM handling.

---

## M15 — Device / Backend Boundary

**Build**

Separate the runtime from the hardware backend.

- Support CPU, PyTorch CUDA, and PyTorch MPS
- `Scheduler`, `Request`, and lifecycle code stay backend-agnostic
- `ModelRunner` owns all device-specific concerns

Out of scope: custom GPU kernels.

**Learn**

- Scheduling and hardware backends are almost fully orthogonal.

---

## M16 — First Custom GPU Kernel

**Build**

Replace exactly one operator with a custom GPU kernel. Start with RMSNorm or a simple elementwise op, not attention.

- Metal or MLX on Apple Silicon; Triton on CUDA
- Keep the PyTorch reference implementation
- Validate numerical correctness and benchmark against PyTorch
- Integrate behind the existing backend boundary

Out of scope: rewriting the full model as custom kernels.

**Learn**

- The path from model → operator → kernel → GPU.
- Kernel launches, memory access patterns, tiles, warps or SIMD groups, and memory bandwidth.

---

## M17 — Paged KV Cache + Paged Attention

**Build**

Replace contiguous KV allocation with fixed-size blocks (for example, 16 tokens per block), and teach attention to read from them.

```text
Request A: block 1, block 4, block 8
```

- A free block pool
- Per-request logical block tables instead of contiguous buffers
- Blocks allocated as sequences grow and released when they finish
- Paged attention in plain PyTorch: look up each request's block table, gather its K/V from the physical blocks, then run attention
- Paged attention replaces the shared attention module from M10, so the Qwen3 layer code does not change
- Block tables travel to the model through `ForwardBatch`
- Validate output against the contiguous KV cache from M14

Out of scope: prefix sharing, fused paged attention kernels such as FlashInfer or vLLM's PagedAttention.

**Learn**

- Fragmentation, allocation, and freeing.
- Block tables and the mapping from logical to physical KV positions.
- Memory layout and attention are coupled: changing where K/V live means changing how attention reads them.
- Why production engines use paged KV memory, and why they need dedicated kernels to avoid the gather cost.

---

## M18 — Second Model: GPT-2

**Build**

Add GPT-2 Small as a second model to prove that the runtime is model-agnostic.

- A GPT-2 implementation registered in the model registry, with its own `ModelConfig`
- GPT-2-specific pieces: learned position embedding, LayerNorm with bias, GELU MLP, and multi-head attention (`num_kv_heads = num_heads`)
- Load Hugging Face weights, including transposing the `Conv1D` weights into regular linear layers
- Reuse the shared paged attention module from M17
- The scheduler rejects requests longer than `max_context_len` (1024 for GPT-2) at admission
- Validate logits against the Hugging Face GPT-2 model
- Success condition: only the new model file, its `ModelConfig`, and its tokenizer are added; `Engine`, `Scheduler`, `Sampler`, `KVCacheManager`, and paged attention do not change

Out of scope: running both models in one engine at the same time.

**Learn**

- Where the model boundary actually sits, tested by a real second model instead of assumed.
- A different position encoding and attention layout only change the model code and `ModelConfig`.
- Model limits such as maximum context length are runtime concerns, because admission depends on them.

---

## M19 — Prefix Cache

**Build**

Reuse KV across requests that share a prompt prefix, using a hash-based lookup.

```text
A: system prompt + document + question A
B: system prompt + document + question B
         └── shared, reuse KV ──┘
```

**Learn**

- The KV cache is not only request-local; it can be reusable computation across requests.

---

## M20 — Radix Cache

**Build**

Replace the hash-based prefix cache with a radix tree that manages variable-length prefixes.

Split the scheduler's policy from its mechanism:

- Mechanism stays in `Scheduler`: waiting and running queues, KV capacity checks, block allocation, and building `ScheduleBatch`
- Policy moves behind a `SchedulePolicy` interface that orders the waiting queue, for example `order(waiting) -> list[Request]`
- At least two policies, selected by configuration: FCFS and longest-prefix-match, which runs requests with the most cached prefix first
- Compare cache hit rate and TTFT between the two policies on the same workload

**Learn**

- Prefix matching, cache organization, and cache-aware scheduling.
- Separating policy from mechanism lets the scheduling algorithm change without touching queue or memory management.
- Why SGLang's RadixAttention / radix cache is valuable.

---

## M21 — Chunked Prefill

**Build**

Split very long prompts (for example, 100k tokens) into chunks so decode requests can run in between.

```text
chunk 1 → decode step → chunk 2 → decode step → chunk 3
```

**Learn**

- The scheduler does more than order requests; it shapes the GPU workload.

---

## M22 — Overlap Scheduling

**Build**

Remove the idle gaps between CPU scheduling and GPU execution.

```text
GPU executes step N  ║  CPU prepares step N+1
```

**Learn**

- CPU/GPU pipelining, asynchronous execution, and synchronization.
- How much scheduling overhead costs when it is not hidden.

---

## M23 — Mini Inference Runtime

**Build**

The complete architecture:

```text
                    Engine
                      │
                 Scheduler ──── SchedulePolicy
                /          \
          waiting         running
              │
        ScheduleBatch
              │
          ModelWorker
              │
         ForwardBatch
              │
          ModelRunner ──── Sampler
              │
             Model ──── ModelRegistry
              │
        Attention / MLP
              │
           Backend
              │
             GPU


       KVCacheManager
              │
         Block Manager
              │
        Prefix/Radix Cache


          Profiler
              │
     request/runtime/GPU
```

Out of scope: tensor parallelism, pipeline parallelism, MoE, distributed serving, speculative decoding, grammar-constrained decoding, LoRA, quantization.

**Learn**

- How every layer of a modern inference engine fits together, and why each one exists.
- By around M14, the abstractions in the SGLang scheduler, model runner, and KV cache code should start to feel familiar.

---

## Credits

This roadmap draws on the following projects and courses:

- [tiny-llm](https://skyzh.github.io/tiny-llm/) ([GitHub](https://github.com/skyzh/tiny-llm)) by skyzh: a course on building an LLM serving system with MLX on Apple Silicon. Its Week 1 is the reference for M10.
- [mini-sglang](https://github.com/sgl-project/mini-sglang) by the SGLang team: a compact implementation of SGLang, useful for comparing each milestone against the real design.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): a minimal vLLM-style inference engine in about 1,200 lines of Python.
- [CMU LLM Systems](https://llmsystem.github.io/) (11-868, [course notes on csdiy.wiki](https://csdiy.wiki/en/%E6%B7%B1%E5%BA%A6%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/CMU11-868/)): a graduate course on the full LLM systems stack, from GPU acceleration to distributed training and serving.
