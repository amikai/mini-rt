# mini-rt

A minimal LLM inference runtime, built from scratch one concept at a time.

## Why This Repo Exists

I built this repo to learn what [SGLang](https://github.com/sgl-project/sglang) does and why it is designed the way it is.

Reading a production inference engine directly is hard. Abstractions such as the scheduler, `ScheduleBatch`, `ForwardBatch`, the KV cache pool, and the radix cache all exist to solve specific problems, and the code rarely tells you what those problems were.

So this repo does not clone SGLang feature by feature. It starts from a bare Qwen3 generation loop and runs into the same problems SGLang was built to solve, one at a time, solving each one by hand. Each milestone adds exactly one core idea and answers one new systems question.

After M16, the SGLang scheduler, model runner, and KV cache code should read as optimized versions of things already built here.

## Design Rule

**Same boundaries and names as SGLang, simplest possible mechanism.**

- Component boundaries and names follow SGLang: `Scheduler`, `ScheduleBatch`, `ForwardBatch`, `ForwardMode`, `ModelRunner`, `AttentionBackend`, `TokenToKVPoolAllocator`, `RadixCache`, `SchedulePolicy`. When I later open the SGLang source, each name points at something I already built.
- Inside each component, use the easiest code to understand: Python lists, plain loops, and plain PyTorch.
- A simplification is allowed only if SGLang's version reads as an optimized form of mine. A simplification that teaches a different mental model, one I would have to unlearn, is not allowed.

The simplest version is often one SGLang already has. For example, SGLang's `torch_native` attention backend runs attention one request at a time in a Python loop, reading each request's KV slots from the pool. That is exactly the attention this repo uses from M10 onward.

Each milestone ends with a **Compare with SGLang** line that lists the SGLang files to read next. Paths are relative to `python/sglang/srt/` on SGLang `main` as of September 2026.

## Model

The whole roadmap uses [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). It is small enough to run on a laptop, and its architecture matches the models SGLang serves in production: RoPE, grouped-query attention (GQA), RMSNorm, and a SwiGLU MLP. Using the same model from M1 onward means the tokenizer, EOS tokens, KV cache shape, and position handling never change underneath the runtime while it is being built. M15 then adds GPT-2 as a second model to test that the runtime does not depend on Qwen3.

## Hardware

Every milestone must run on both Apple Silicon (PyTorch MPS) and NVIDIA GPUs (PyTorch CUDA), selected by a single `device` setting. The core roadmap uses only PyTorch operators, so the same code runs on both. The places where the platforms differ are small and explicit:

- Device synchronization before reading a timer (M9)
- The memory budget used to size the KV pool (M14)
- How much CPU/GPU overlap is possible (M18)

Custom kernels (Metal on Apple, Triton on NVIDIA) are optional and live in [Optional Milestones](#optional-milestones).

## Roadmap at a Glance

| Milestone | Question | Answer |
|-----------|----------|--------|
| M1 | How does a model generate tokens? | Autoregressive loop |
| M2 | How do I turn it into a service? | Request + Engine |
| M3 | What if many requests arrive at once? | Scheduler loop |
| M4 | Who is responsible for what? | Layer separation |
| M5 | What is a request over time? | State machine |
| M6 | Running one request at a time wastes the GPU | Static batching |
| M7 | Some requests in a batch finish early | Continuous batching |
| M8 | Each request wants different sampling settings | Per-request sampling |
| M9 | Where is the time actually going? | Profiling |
| M10 | What does the model actually compute? | Own model + packed tokens |
| M11 | Why recompute everything every step? | KV cache + `ForwardMode` |
| M12 | Should scheduling data and execution data be the same? | `ScheduleBatch` / `ForwardBatch` |
| M13 | Who decides how attention reads K/V? | `AttentionBackend` |
| M14 | Who manages KV memory, and what if it runs out? | Token KV pool + admission + retract |
| M15 | Is the runtime really model-agnostic? | Second model: GPT-2 |
| M16 | Why can't requests share computation? | Radix cache + schedule policy |
| M17 | A huge prompt is blocking the GPU | Chunked prefill |
| M18 | The CPU and GPU keep waiting on each other | Overlap scheduling |
| M19 | — | Mini inference runtime |

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

## M2 — Single Request Runtime

**Build**

Turn `generate(prompt)` into `Request → Runtime → Model`.

- A `Req` with `rid`, `prompt`, `input_ids`, `output_ids`, `max_new_tokens`
- An `Engine` that runs one request at a time
- If a request is already running, new requests are rejected as busy

Out of scope: queues, scheduling, batching, caching.

**Learn**

- Serving a model is different from calling a model.
- Request lifecycle, runtime state, capacity, and admission.

## M3 — Scheduler Loop

**Build**

Queue requests instead of rejecting them, and give the scheduler the loop shape that SGLang uses.

```python
while True:
    self.recv_requests()                        # move new requests into waiting_queue
    batch = self.get_next_batch_to_run()        # decide what runs this step
    if batch is None:
        continue
    result = self.run_batch(batch)              # run the model
    self.process_batch_result(batch, result)    # append tokens, check finish, free resources
```

- A `waiting_queue` in FIFO order
- One running request at a time; the next waiting request starts when it finishes
- No model or PyTorch logic inside the scheduler

```text
submit A, B, C  →  execution order: A, B, C
```

This four-step loop never changes shape again. Every later milestone only changes what happens inside one of the four steps.

Out of scope: batching, KV cache, advanced scheduling policies.

**Learn**

- The scheduler decides *who runs*; the model decides *how to compute*.
- The first split between control plane and compute.

**Compare with SGLang**: `event_loop_normal()` in `managers/scheduler.py`.

## M4 — Runtime Layer Separation

**Build**

No new features. Refactor into these components:

```text
Engine
  ├─ Scheduler       decides who runs
  └─ ModelRunner     owns device placement and forward execution (tensor → logits)
       └─ Sampler    turns logits into the next token (greedy)
```

- No model logic in `Scheduler`, no queue logic in `ModelRunner`, no model knowledge in `Sampler`
- Behavior stays identical to M3

**Learn**

- Runtime orchestration, scheduling, model execution, and sampling are separate responsibilities.
- This is the basic shape of SGLang and vLLM.

**Compare with SGLang**: `model_executor/model_runner.py` (`ModelRunner.sample()`), `layers/sampler.py`.

## M5 — Request State Machine

**Build**

Make a request a state machine instead of a plain data object.

```text
WAITING → RUNNING → FINISHED
                  ↘ FAILED
```

- Explicit state transitions in `Scheduler` and `Engine`
- Termination by EOS token, `max_new_tokens`, or exception, recorded as a finish reason
- Support more than one EOS token ID (Qwen3 uses both `<|im_end|>` and `<|endoftext|>`)

Out of scope: batching, KV cache.

**Learn**

- An inference request is a long-lived state machine, not a function call.
- This is the foundation for continuous batching, cancellation, streaming, and timeouts.

**Compare with SGLang**: `Req` and the `FINISH_*` classes in `managers/schedule_batch.py`.

## M6 — Static Batching

**Build**

Serve many requests with one model forward pass.

- `get_next_batch_to_run()` picks up to `max_batch_size` waiting requests
- Padded batch tensor with attention masks for different sequence lengths (the Hugging Face model needs this layout)
- A batch stays together until every request in it finishes; empty slots are not refilled

Out of scope: continuous batching, KV cache.

**Learn**

- Why GPU throughput depends on batching.
- How to handle mismatched sequence lengths with padding, masks, and a batch dimension.

## M7 — Continuous Batching

**Build**

Refill a batch slot as soon as a request finishes.

```text
A B C  →  _ B C  →  D B C
```

- Separate `waiting_queue` and `running_batch`
- Every loop iteration re-decides what runs
- Finished requests leave `running_batch` immediately
- Padded tensors are rebuilt every step, and the full sequence is still recomputed

Out of scope: KV cache.

**Learn**

- Continuous batching is a scheduling problem, not a Transformer problem.
- The scheduler's real job is to rebuild the workload on every step.

**Compare with SGLang**: `filter_batch()` and `merge_batch()` in `managers/schedule_batch.py`.

## M8 — Per-Request Sampling

**Build**

Replace greedy-only decoding with sampling parameters that belong to each request.

- `SamplingParams` on each `Req`: `temperature`, `top_k`, `top_p`, and an optional seed
- `temperature = 0` means greedy, so M1–M7 behavior is still reachable
- The `Sampler` handles one batch where every row can have different parameters
- Parameters are turned into per-row tensors, so sampling runs as batched tensor ops rather than a Python loop over requests

Out of scope: repetition penalties, logit bias, constrained or grammar-based decoding.

**Learn**

- Sampling settings are request state, not model state.
- Continuous batching mixes requests with different settings in the same batch, so the sampler must be batch-aware.
- How temperature, top-k, and top-p reshape the probability distribution.

**Compare with SGLang**: `sampling/sampling_params.py`, `sampling/sampling_batch_info.py`.

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

- Synchronize the device before reading a timer: `torch.mps.synchronize()` on Apple, `torch.cuda.synchronize()` on NVIDIA
- Optional operator-level profiling with `torch.profiler`; on Apple, also `torch.mps.profiler` with the Metal System Trace in Xcode Instruments

**Learn**

- The core engineering loop: measure → understand → optimize → measure again.
- CPU wall-clock latency is not GPU execution time; asynchronous GPU execution must be synchronized before timing.

## M10 — Own Model + Packed Tokens

**Build**

Replace the Hugging Face model with a hand-written Qwen3. Load only the pretrained weights from Hugging Face.

Model pieces:

- Token embedding and tied LM head
- RMSNorm
- Rotary position embedding (RoPE)
- Grouped-query attention (GQA): 16 query heads share 8 KV heads
- Per-head RMSNorm on Q and K, as Qwen3 does
- SwiGLU MLP
- A `ModelConfig` with `num_layers`, `num_heads`, `num_kv_heads`, `head_dim`, `vocab_size`, `max_context_len`, `eos_token_ids`, and `dtype`
- A model registry that selects the model implementation and its `ModelConfig` by name

Packed input instead of padding:

- All tokens of all requests in the batch are concatenated into one 1D `input_ids` tensor, with a matching 1D `positions` tensor and a per-request `seq_lens` list
- Embedding, linear layers, norms, and the MLP run on all tokens at once, because they work on each token independently
- Attention is the only operation that needs request boundaries, so it loops over requests in Python and calls `torch.nn.functional.scaled_dot_product_attention` once per request
- No padding and no attention mask

```text
input_ids: [a1 a2 a3 | b1 b2 | c1 c2 c3 c4]
seq_lens:  [3, 2, 4]
```

Validation:

- Validate logits against the Hugging Face model within a tolerance
- Re-run the M9 benchmarks as the new baseline

Out of scope: KV cache, custom kernels, a second model architecture (see M15).

**Learn**

- A model is a fixed sequence of matrix operations with known shapes and costs.
- Attention is the only operation that mixes tokens, which is why packing is easy: only attention needs to know where each request starts and ends.
- How GQA sets the size of the KV cache, and how RoPE depends on correct token positions.
- The runtime can only change how attention reads K/V if it owns the model code.

**Compare with SGLang**: `_run_sdpa_forward_extend()` in `layers/attention/torch_native_backend.py` loops over requests the same way.

Reference: [tiny-llm](https://skyzh.github.io/tiny-llm/) Week 1 builds the same Qwen3 pieces step by step.

## M11 — KV Cache + ForwardMode

**Build**

Remove the redundant computation that M9 exposed.

- Per-request KV cache: each request owns one K tensor and one V tensor per layer, and each new token appends to them
- A `ForwardMode` enum with two values:
  - `EXTEND`: run the whole prompt of new requests and fill their KV cache
  - `DECODE`: run one new token per request and append to the existing cache
- One forward path for both modes. Only attention looks at the mode.
- A batch is either all `EXTEND` or all `DECODE`. `get_next_batch_to_run()` runs a prefill batch when new requests are waiting, and otherwise runs a decode batch for `running_batch`.
- Output is validated against the non-cached implementation
- Re-run the M9 benchmarks and compare forward latency, ITL, processed tokens, and compute amplification against M10

SGLang calls prefill `EXTEND` because, once prefix caching exists (M16), a new request extends a cached prefix instead of always starting from zero. This repo uses the same name from the start.

Out of scope: memory pools, prefix sharing, mixing extend and decode in one batch.

**Learn**

- Why the KV cache exists: the causal mask means K/V for past tokens never change.
- Model execution now carries persistent state across steps.
- Extend and decode have very different GPU characteristics: extend is compute-bound, decode is memory-bound.

**Compare with SGLang**: `ForwardMode` in `model_executor/forward_batch_info.py`, `get_next_batch_to_run()` in `managers/scheduler.py`.

## M12 — ScheduleBatch / ForwardBatch

**Build**

Split the CPU-side scheduling representation from the device-side execution representation.

- `ScheduleBatch`: Python `Req` objects, `forward_mode`, and scheduling metadata
- `ForwardBatch`: only tensors and plain values that the model needs: `forward_mode`, `input_ids`, `positions`, `seq_lens`, and cache metadata
- `ForwardBatch.init_new(schedule_batch, model_runner)` is the single conversion point
- `ModelRunner` only sees `ForwardBatch`; it never touches a `Req`

```text
Scheduler → ScheduleBatch
──────── ForwardBatch.init_new() ────────
ModelRunner → ForwardBatch → GPU
```

No new optimization behavior.

**Learn**

- Why control-plane data structures and execution data structures should be separate.
- This maps directly onto SGLang's internals.

**Compare with SGLang**: `ScheduleBatch` in `managers/schedule_batch.py`, `ForwardBatch` in `model_executor/forward_batch_info.py`, and the `ForwardBatch.init_new()` call in `managers/tp_worker.py`.

## M13 — AttentionBackend

**Build**

Move everything about how attention reads and writes K/V out of the model code.

- An `AttentionBackend` with one method: `forward(q, k, v, layer_id, forward_batch)`
- The backend writes the new K/V into the cache, then computes attention, branching on `forward_batch.forward_mode`
- The Qwen3 attention layer computes Q, K, V, applies RoPE and Q/K norm, then calls the backend. It no longer knows where the cache lives.
- One implementation, `TorchNativeBackend`: the per-request loop from M10 and the per-request cache from M11, moved behind the interface
- The backend is created by `ModelRunner` and travels to the layers through `ForwardBatch`

No behavior change. The test for this milestone is M14: replacing the whole KV storage must only change the backend, not the model.

Out of scope: more than one backend, custom kernels.

**Learn**

- Memory layout and attention are coupled: whoever decides where K/V live must also decide how attention reads them.
- Which parts are model-specific (embedding, norm, MLP, position encoding) and which parts the runtime owns (attention over cached K/V).

**Compare with SGLang**: `layers/radix_attention.py` (the model-side layer), `layers/attention/base_attn_backend.py` (`forward`, `forward_extend`, `forward_decode`).

## M14 — Token KV Pool + Admission + Retract

**Build**

Move KV ownership from the request to the runtime, and handle running out of memory.

The pool:

- One preallocated K buffer and one V buffer per layer, each shaped `[num_slots, num_kv_heads, head_dim]`. One slot holds the K/V of one token.
- `num_slots` comes from the memory budget: free device memory after loading weights, times a fraction such as 0.8. Use `torch.cuda.mem_get_info()` on NVIDIA; on Apple, use `torch.mps.recommended_max_memory()` minus current allocation, and leave room for the rest of the system because memory is shared with the CPU.
- A `TokenToKVPoolAllocator` that keeps free slots in a Python list: `alloc(n) -> list[int]` and `free(slots)`
- Each `Req` keeps `kv_indices: list[int]`, the slots that hold its tokens, in order
- `ForwardBatch` carries `out_cache_loc`, the slots where this step's new tokens are written, and each request's `kv_indices`
- Only `TorchNativeBackend` changes: it writes K/V with `k_buf[out_cache_loc] = k` and gathers each request's K/V with `k_buf[kv_indices]`

The scheduler owns the allocator, like SGLang:

- Admission: a waiting request starts only if `free slots >= prompt length + a reserve for future decode tokens`
- Every step allocates slots for the new tokens before building `ForwardBatch`
- Finished requests free their slots in `process_batch_result()`
- Retract: if a decode step cannot get enough slots, move the most recent running request back to `waiting_queue` and free its slots. It is recomputed from its prompt and generated tokens when it is admitted again.

```text
free slots: [0 1 2 3 4 5 6 7 8 9 ...]
Req A kv_indices: [0 1 2 7]
Req B kv_indices: [3 4 5 6 8]
```

Out of scope: blocks larger than one token (see Optional Milestones), prefix sharing, smarter retraction policies.

**Learn**

- Request lifecycle and memory lifecycle are different things.
- With one token per slot, there is no fragmentation and no partially filled block to track. A request's slots do not need to be contiguous.
- Admission and retraction are scheduler decisions driven by memory, not by batch size.

**Compare with SGLang**: `ReqToTokenPool` and `MHATokenToKVPool` in `mem_cache/memory_pool.py` (`ReqToTokenPool` is the tensor form of every request's `kv_indices`), `TokenToKVPoolAllocator` in `mem_cache/allocator/token.py`, `prepare_for_extend()`, `prepare_for_decode()`, and `retract_decode()` in `managers/schedule_batch.py`, and `PrefillAdder` in `managers/schedule_policy.py`.

## M15 — Second Model: GPT-2

**Build**

Add GPT-2 Small as a second model to prove that the runtime is model-agnostic.

- A GPT-2 implementation registered in the model registry, with its own `ModelConfig`
- GPT-2-specific pieces: learned position embedding, LayerNorm with bias, GELU MLP, and multi-head attention (`num_kv_heads = num_heads`)
- Load Hugging Face weights, including transposing the `Conv1D` weights into regular linear layers
- Reuse `TorchNativeBackend` and the KV pool from M14
- The scheduler rejects requests longer than `max_context_len` (1024 for GPT-2) at admission
- Validate logits against the Hugging Face GPT-2 model
- Success condition: only the new model file, its `ModelConfig`, and its tokenizer are added; `Engine`, `Scheduler`, `Sampler`, the KV pool, and the attention backend do not change

Out of scope: running both models in one engine at the same time.

**Learn**

- Where the model boundary actually sits, tested by a real second model instead of assumed.
- A different position encoding and attention layout only change the model code and `ModelConfig`.
- Model limits such as maximum context length are runtime concerns, because admission depends on them.

## M16 — Radix Cache + Schedule Policy

**Build**

Reuse KV across requests that share a prompt prefix.

```text
A: system prompt + document + question A
B: system prompt + document + question B
         └── shared, reuse KV ──┘
```

The cache:

- A `RadixCache`: a tree where each edge holds a run of token IDs and the KV slots for those tokens. With one token per slot, a prefix can end at any token.
- `match_prefix(token_ids) -> (kv_indices, last_node)`: the longest cached prefix of a new request
- When a request finishes, insert its tokens and slots into the tree instead of freeing them
- A lock count on each node: a node that a running request is using cannot be evicted
- When the allocator runs out of free slots, evict unlocked leaves in least-recently-used order and return their slots to the allocator
- An `EXTEND` batch now computes only the tokens after the matched prefix; attention still reads the prefix slots

The policy:

- Split the scheduler's policy from its mechanism
- Mechanism stays in `Scheduler`: queues, admission, allocation, building `ScheduleBatch`
- Policy moves behind a `SchedulePolicy` that orders `waiting_queue`
- Two policies, selected by configuration: FCFS and longest-prefix-match (LPM), which runs requests with the longest cached prefix first
- Compare cache hit rate and TTFT between the two policies on the same workload

Out of scope: hash-based block prefix caching (vLLM's approach), cache offloading to CPU memory.

**Learn**

- The KV cache is not only request-local; it can be reusable computation across requests.
- Why a radix tree fits variable-length prefixes, and how eviction and locking interact with allocation.
- Separating policy from mechanism lets the scheduling algorithm change without touching queue or memory management.
- Why SGLang's RadixAttention is valuable.

**Compare with SGLang**: `mem_cache/radix_cache.py` (`match_prefix`, `insert`, `evict`, `inc_lock_ref`), `SchedulePolicy` in `managers/schedule_policy.py`.

## M17 — Chunked Prefill

**Build**

Split very long prompts into chunks so decode requests can run in between.

```text
chunk 1 → decode step → chunk 2 → decode step → chunk 3
```

- A `chunked_prefill_size` token budget for each `EXTEND` batch
- A prompt longer than the budget is extended one chunk per step; the scheduler keeps it as `chunked_req` until its last chunk
- Each chunk is an ordinary `EXTEND` whose prefix is the chunks already computed, so M16's prefix handling already covers it
- Simple interleaving rule: after each chunk, run one decode step for `running_batch`
- Measure ITL for running requests while a long prompt arrives, with and without chunking

**Learn**

- The scheduler does more than order requests; it shapes the GPU workload.
- Why a token budget per step, not a request count, is what the scheduler really controls.

**Compare with SGLang**: `chunked_req` in `managers/scheduler.py`, the chunk budget in `PrefillAdder` (`managers/schedule_policy.py`). SGLang's interleaving rule is more complex than the one used here.

## M18 — Overlap Scheduling

**Build**

Remove the idle gaps between CPU scheduling and GPU execution.

```text
GPU executes step N  ║  CPU prepares step N+1
```

- Find and remove sync points on the scheduling path, such as `.item()`, `.cpu()`, and `.tolist()` on GPU tensors
- Keep the sampled `next_token_ids` on the device, and use that tensor directly as the `input_ids` of the next decode step, so step N+1 can be queued before step N finishes
- Move `process_batch_result()` for step N to after step N+1 has been queued. EOS and `max_new_tokens` are now detected one step late; drop the one extra token and free the request then.
- Allocate KV slots for step N+1 before knowing whether a request finished in step N
- **Apple**: MPS queues GPU work asynchronously, so this ordering alone creates the overlap
- **NVIDIA**: the same approach first, then optionally CUDA streams and events for explicit overlap
- Measure GPU idle time between steps on both platforms

**Learn**

- CPU/GPU pipelining, asynchronous execution, and synchronization.
- The hard part of overlap is data dependence: the next step's input is the current step's output.
- How much scheduling overhead costs when it is not hidden.

**Compare with SGLang**: `event_loop_overlap()` in `managers/scheduler.py`, `FutureMap` in `managers/overlap_utils.py`, which generalizes "use the output tensor as the next input".

## M19 — Mini Inference Runtime

**Build**

The complete architecture:

```text
Engine
  │
Scheduler ─────────────── SchedulePolicy (FCFS / LPM)
  │  waiting_queue
  │  running_batch
  │  chunked_req
  ├─ RadixCache
  │     └─ TokenToKVPoolAllocator (free slots)
  │
ScheduleBatch
  │
──── ForwardBatch.init_new() ────
  │
ForwardBatch (forward_mode, input_ids, positions, seq_lens, out_cache_loc, kv_indices)
  │
ModelRunner ───────────── Sampler
  │
Model (Qwen3 / GPT-2) ─── ModelRegistry
  │
AttentionBackend ──────── KV pool (K/V buffers per layer)
  │
Apple GPU / NVIDIA GPU    (Profiler measures every layer above)
```

SGLang also has a `TpModelWorker` between the scheduler and `ModelRunner`, which exists for tensor parallelism. This repo runs on one device, so `Scheduler` calls `ModelRunner` directly.

Out of scope: tensor parallelism, pipeline parallelism, MoE, distributed serving, speculative decoding, grammar-constrained decoding, LoRA, quantization.

**Learn**

- How every layer of a modern inference engine fits together, and why each one exists.
- The SGLang scheduler, model runner, and KV cache code should now read as optimized versions of these components.

## Optional Milestones

These are not needed to understand SGLang's design. Each one goes deeper into a single topic and can be done after the milestone it depends on.

**Custom RMSNorm kernel** (after M10). Replace one operator with a hand-written kernel: Metal on Apple via `torch.mps.compile_shader`, Triton on NVIDIA. Validate against the PyTorch version and benchmark it. Learn the path from operator to kernel, and why a fused kernel beats a chain of PyTorch ops.

**Triton attention backend** (after M14). A second `AttentionBackend` whose decode kernel reads K/V directly through `kv_indices`, without the gather step. Compare with `layers/attention/triton_backend.py`.

**Blocks larger than one token** (after M16). Make the slot size a `page_size` setting and support 16 tokens per block. Learn what gets harder: partially filled blocks, and prefix matching that must align to block boundaries.

**CUDA graphs** (after M18, NVIDIA only). Capture decode steps for a fixed set of batch sizes and replay them. For a 0.6B model, decode time is mostly kernel launch overhead, so this is one of the largest speedups. Compare with `model_executor/runner/base_cuda_graph_runner.py`.

**Process split + streaming** (after M9). Move tokenization and detokenization out of the scheduler loop into separate processes, add an HTTP API, and stream tokens with incremental detokenization. Compare with `managers/tokenizer_manager.py` and `managers/detokenizer_manager.py`.

## Credits

This roadmap draws on the following projects and courses:

- [tiny-llm](https://skyzh.github.io/tiny-llm/) ([GitHub](https://github.com/skyzh/tiny-llm)) by skyzh: a course on building an LLM serving system with MLX on Apple Silicon. Its Week 1 is the reference for M10.
- [mini-sglang](https://github.com/sgl-project/mini-sglang) by the SGLang team: a compact implementation of SGLang, useful for comparing each milestone against the real design.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): a minimal vLLM-style inference engine in about 1,200 lines of Python.
- [CMU LLM Systems](https://llmsystem.github.io/) (11-868, [course notes on csdiy.wiki](https://csdiy.wiki/en/%E6%B7%B1%E5%BA%A6%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/CMU11-868/)): a graduate course on the full LLM systems stack, from GPU acceleration to distributed training and serving.
