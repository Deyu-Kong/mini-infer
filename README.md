# mini-infer

> A lightweight C++/CUDA LLM inference engine built from scratch, supporting Qwen2.5 series models with PagedAttention, speculative decoding, and prefix caching. Includes an OpenAI-compatible HTTP serving mode and a Python SDK that work both **offline** (local GPU) and **online** (client/server).

## Architecture

![Architecture](docs/architecture.drawio.svg)

## Supported Models

The engine auto-detects architecture from HuggingFace `config.json` and dispatches
via `ArchRegistry`. Any model sharing the **RMSNorm + RoPE + SwiGLU + GQA** family
works out of the box:

| Architecture | Models | Status |
|-------------|--------|--------|
| QwenLLaMA | Qwen2/2.5, LLaMA 2/3/3.1, Mistral, Yi, DeepSeek, Phi3 | ✅ verified |
| Gemma | Gemma 1/2/3 (GeGLU, 4-norm block, Q/K RMSNorm, sliding window) | ✅ verified |
| MoE | Mixtral, DeepSeek-MoE, Qwen2-MoE (with shared expert) | ✅ verified |

**Primary test model**: Qwen2.5-7B-Instruct / Qwen2.5-Coder-7B-Instruct (FP16)
**Draft model** (speculative decoding): Qwen2.5-Coder-1.5B-Instruct (FP16)

## Hardware

4 × RTX A6000 (48GB each), CUDA 12.x, sm_86 (Ampere).

## Build

```bash
scripts/build.sh        # configure + build
scripts/run_tests.sh    # run all tests via ctest
```

Or manually:

```bash
export PATH=/data1/kdy/anaconda3/envs/vllm/bin:/usr/local/cuda-12.1/bin:/data1/tyh/miniconda3/bin:$PATH
export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 CUDAHOSTCXX=/usr/bin/g++-9
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j
cd build && ctest --output-on-failure
```

## Quick Start

### 1. Build & Test (30 sec)

```bash
scripts/build.sh          # CMake + Ninja Release build
scripts/run_tests.sh      # 15+ unit tests, all kernel + layer + model
```

### 2. CLI — offline & online

mini-infer has one binary, three subcommands: `generate` (offline, default),
`serve` (online server), and `client` (online client). The legacy
`mini_infer --model ... --prompt ...` form still works (offline).

```bash
# Offline: one-shot inference (loads model, generates, exits)
./build/mini_infer generate --model /path/to/Qwen2.5-7B-Instruct \
    --prompt "Explain quantum computing" --max-new-tokens 100 --greedy

# Offline: PagedAttention / Speculative decoding (same as before)
./build/mini_infer --model /path/to/Qwen2.5-7B-Instruct \
    --prompt "Write a Python function" --max-new-tokens 100 --greedy --paged
./build/mini_infer --model /path/to/Qwen2.5-7B-Instruct \
    --spec-draft /path/to/Qwen2.5-Coder-1.5B-Instruct \
    --prompt "Explain transformers" --max-new-tokens 100 --greedy --gamma 4
```

**Online service** — `serve` loads the model **once** into GPU memory and
exposes a JSON + OpenAI-compatible HTTP API; every request reuses the loaded
weights (no per-call reload), which is the key win over the offline path:

```bash
# Terminal 1: start the server
./build/mini_infer serve --model /path/to/Qwen2.5-7B-Instruct --port 8000

# Terminal 2: query it with the C++ client (non-stream or --stream)
./build/mini_infer client --remote http://localhost:8000 \
    --prompt "Explain quantum computing" --max-tokens 100 --greedy
./build/mini_infer client --remote http://localhost:8000 --prompt "Count 1..5" --stream

# Or with curl / any OpenAI client
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}'
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /generate`,
`POST /v1/chat/completions` (OpenAI-compatible, `stream:true` → SSE),
`POST /tokenize`, `POST /detokenize`.

### 3. Python SDK — offline & online, one API

```bash
pip install -e sdk/   # install mini-infer-sdk
```

```python
from mini_infer_sdk import MiniInfer

# Offline: wraps the C++ binary (one subprocess per call)
engine = MiniInfer(model_path="/path/to/Qwen2.5-7B-Instruct")

# Online: connects to a running `mini_infer serve` (persistent model)
engine = MiniInfer(endpoint="http://localhost:8000")

# Same API either way:
result = engine.generate("Explain the transformer architecture.")
print(result.text, f"{result.tokens_per_sec:.1f} tok/s")

# Multi-turn chat with history
chat = engine.chat(system_prompt="You are a concise AI assistant.")
chat.send("What is the capital of France?")
chat.send("What is its population?")          # remembers context

# Streaming (SSE for online; line-buffered for offline)
for chunk in engine.stream("Count from 1 to 5."):
    print(chunk, end="", flush=True)

# Batch processing
batch = engine.batch()
stats = batch.process(["Summarize Python.", "Sort a list.", "Explain GIL."])
print(f"Throughput: {stats.tokens_per_sec:.1f} tok/s")
```

The backend is auto-selected from the constructor arg (`model_path` →
`LocalBackend`, `endpoint` → `RemoteBackend`); `ChatSession`/`BatchInfer`
work unchanged in both modes.

**Online service from the SDK** — `MiniInferServer` launches and manages a
persistent service (the C++ `serve` binary, model loaded once and resident);
the SDK client then sends a continuous stream of requests to it:

```python
from mini_infer_sdk import MiniInfer, MiniInferServer

with MiniInferServer(model="/path/to/model", port=8000) as srv:
    engine = MiniInfer(endpoint=srv.endpoint)   # client to the resident model
    for q in questions:                         # continuous requests
        print(engine.generate(q).text)
# service torn down on exit. See sdk/examples/05_online_service.py for a
# sequential + concurrent traffic demo. No --python needed: the SDK hands
# its own interpreter (which has `tokenizers`) to the engine.
```

```bash
# Unified Python CLI (offline + online)
python -m mini_infer_sdk generate --model /path/to/model --prompt "Hi"
python -m mini_infer_sdk generate --endpoint http://localhost:8000 --prompt "Hi"
python -m mini_infer_sdk chat --model /path/to/model        # interactive REPL
python -m mini_infer_sdk serve --model /path/to/model --port 8000
```

### 4. Run benchmarks

```bash
# Kernel microbenchmarks
./build/benchmarks/bench_kernels

# Static batching benchmark
./build/benchmarks/bench_static \
    --model /path/to/Qwen2.5-7B-Instruct \
    --dataset benchmarks/datasets/sharegpt_sample.json

# Continuous batching benchmark (2.14x throughput)
./build/benchmarks/bench_continuous \
    --model /path/to/Qwen2.5-7B-Instruct \
    --dataset benchmarks/datasets/sharegpt_sample.json
```

## CLI Reference

The binary dispatches on the first argument: `serve` (online server),
`client` (online client), `generate` or no subcommand (offline, default).

### Offline — `mini_infer generate` (default)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model DIR` | string | *required* | HuggingFace model directory |
| `--prompt TEXT` | string | `"你好，请介绍一下你自己。"` | Input prompt |
| `--max-new-tokens N` | int | `100` | Max tokens to generate |
| `--max-seq-len M` | int | `2048` | Max sequence length |
| `--device D` | int | `0` | GPU device index |
| `--greedy` | flag | — | Greedy sampling (default: top-p) |
| `--temperature T` | float | `1.0` | Sampling temperature |
| `--top-p P` | float | `0.9` | Nucleus sampling probability |
| `--raw` | flag | — | Use prompt verbatim (skip ChatML wrap) |
| `--python PY` | string | `python3` | Python executable for the tokenizer |
| `--paged` | flag | — | Enable PagedAttention |
| `--seed S` | int | `42` | Random seed |
| `--spec-draft DIR` | string | — | Draft model for speculative decoding |
| `--gamma G` | int | `4` | Speculative decoding gamma |

### Online — `mini_infer serve`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model DIR` | string | *required* | Model directory (loaded once, kept in GPU memory) |
| `--draft DIR` | string | — | Draft model for speculative decoding |
| `--host H` | string | `0.0.0.0` | Listen address |
| `--port P` | int | `8000` | Listen port |
| `--device D` | int | `0` | GPU device index |
| `--max-seq-len M` | int | `2048` | Max sequence length |
| `--max-new-tokens N` | int | `512` | Default per-request generation cap |
| `--gamma G` | int | `4` | Speculative decoding gamma |
| `--python PY` | string | `python3` | Python executable for the tokenizer |

### Online — `mini_infer client`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--remote URL` | string | *required* | Server URL, e.g. `http://host:8000` |
| `--prompt TEXT` | string | `"Hello!"` | User message |
| `--system TEXT` | string | — | Optional system prompt |
| `--max-tokens N` | int | `256` | Max tokens to generate |
| `--temperature T` | float | `1.0` | Sampling temperature |
| `--top-p P` | float | `0.9` | Nucleus sampling probability |
| `--greedy` | flag | — | Greedy sampling |
| `--stream` | flag | — | Stream the response (SSE) as it arrives |

## Key Features

### PagedAttention

Block-table based KV cache management inspired by vLLM's virtual memory approach. Each block holds 16 tokens of K/V vectors; logical-to-physical address mapping eliminates memory fragmentation.

- Memory utilization: 60% → 95%
- Max concurrent sequences: 4x improvement

### Speculative Decoding

Draft model (0.5B/1.5B) generates γ candidate tokens, target model (7B) verifies all in one forward pass. Accept/reject with `min(1, p_target/p_draft)` preserves output distribution.

- Greedy output matches naive autoregressive exactly
- Acceptance rate: 88-93% (Qwen2.5-Coder-7B + 1.5B draft)
- Decode speedup: 2-3x

### Prefix Caching

Radix Trie indexed by FNV-1a block hashes with LRU eviction and copy-on-write semantics. Shared prefixes (e.g., system prompts) are computed once and reused.

- TTFT reduction: 45% on shared-prefix workloads

### Continuous Batching

Dynamic scheduler allows requests to join/leave at any step, improving GPU utilization over static batching.

- Throughput: 2.14x over static batching (ShareGPT 1000 samples)

## Ablation Experiments (E0-E6)

Qwen2.5-Coder-7B-Instruct + Qwen2.5-Coder-1.5B-Instruct (draft), RTX A6000:

| Experiment | Configuration | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | vs E0 |
| ---------- | ------------- | --------- | --------- | ------------------ | ----- |
| E0 | Naive autoregressive | 120 | 45 | 22.2 | 1.00x |
| E1 | E0 + PagedAttention | 115 | 43 | 23.3 | 1.05x |
| E2 | E1 + Continuous batching | 85 | 35 | 28.6 | 1.29x |
| E3 | E2 + Speculative decoding γ=4 | 82 | 18 | 55.6 | **2.50x** |
| E4 | E2 + Speculative decoding γ=8 | 80 | 15 | 66.7 | **3.00x** |
| E5 | E3 + Prefix Caching | 45 | 18 | 55.6 | 2.50x (TTFT -45%) |
| E6 | E3 + Tree speculation (optional) | - | - | - | - |

Run all experiments: `python benchmarks/ablation/run_all.py --model <target> --draft <draft>`

## Kernel Latency (RTX A6000, FP16)

| Kernel | Shape | median (us) | p99 (us) |
| ------ | ----- | ----------- | -------- |
| RMSNorm | N=4, D=3584 | 6.14 | 8.19 |
| RoPE | B=4, S=512, H=28, D=128 | 67.58 | 83.97 |
| Softmax | N=57344, D=512 | 176.13 | 191.49 |
| SwiGLU | N=2048, I=18944 | 348.16 | 349.18 |

## Project Structure

```
mini-infer/
├── src/
│   ├── core/              # Tensor, Allocator, Engine, Tokenizer, CLI
│   │   ├── main.cc            # subcommand dispatch: serve / client / generate
│   │   ├── server.{h,cc}      # HTTP server + online client (serve/client)
│   │   ├── http_server.{h,cc} # minimal dependency-free HTTP/1.1 + SSE server
│   │   ├── http_client.{h,cc} # minimal socket HTTP client + SSE reader
│   │   ├── engine.{h,cpp}     # autoregressive / paged / batched generate
│   │   └── tokenizer.{h,cpp}
│   ├── kernels/           # Hand-written CUDA kernels
│   │   ├── rmsnorm_kernel.cu
│   │   ├── rope_kernel.cu
│   │   ├── softmax_kernel.cu
│   │   ├── swiglu_kernel.cu
│   │   ├── naive_attn_kernel.cu
│   │   ├── paged_attn_kernel.cu
│   │   ├── sampling_kernel.cu    # greedy, top-p, accept/reject
│   │   └── model_utils_kernel.cu
│   ├── layers/            # MLP, Attention, RMSNorm, RoPE wrappers
│   ├── model/             # QwenModel, SafeTensors loader, ModelConfig
│   ├── scheduler/         # KV Cache management + scheduling
│   │   ├── kv_cache.cpp           # Naive contiguous KV Cache
│   │   ├── paged_kv_cache.cpp     # PagedAttention block table + rollback
│   │   ├── prefix_cache.cpp       # Radix Trie + LRU + CoW
│   │   └── scheduler.cpp          # Continuous batching scheduler
│   └── speculative/       # Speculative decoding
│       ├── draft_engine.cpp       # Draft model (0.5B/1.5B) inference
│       └── spec_decoder.cpp       # generate-verify-accept/reject loop
├── benchmarks/
│   ├── ablation/          # E0-E6 ablation experiment scripts
│   ├── bench_kernels.cpp  # Kernel latency microbenchmark
│   ├── bench_static.cpp   # Static batching benchmark
│   └── bench_continuous.cpp # Continuous batching benchmark
├── tests/                 # 15 unit tests + integration tests
├── sdk/                   # Python SDK (offline + online, one API)
│   ├── mini_infer_sdk/    # MiniInfer facade + backends
│   │   ├── engine.py          # MiniInfer: model_path=offline, endpoint=online
│   │   ├── server.py          # MiniInferServer: managed persistent service
│   │   ├── backends/          # LocalBackend (subprocess), RemoteBackend (HTTP+SSE)
│   │   ├── chat.py            # ChatSession (works in both modes)
│   │   ├── batch.py           # BatchInfer (works in both modes)
│   │   ├── __main__.py        # unified CLI: `python -m mini_infer_sdk`
│   │   └── config.py
│   ├── examples/          # quickstart, chat, batch, benchmark, online service
│   └── setup.py           # pip install -e sdk/
├── docs/
│   ├── blog.md            # Technical blog (3000+ words)
│   └── PROJECT_PLAN.md    # Project plan
└── CMakeLists.txt         # 6 static libraries + 1 executable
```

## Tech Blog

See [`docs/blog.md`](docs/blog.md) for three technical deep-dives:

1. **PagedAttention** — from contiguous memory to block table indirection
2. **KV Cache Rollback** — causal mask bugs and cache synchronization in speculative decoding
3. **Prefix Caching × Speculative Decoding** — Radix Trie + CoW synergy

## References

1. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
2. Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
3. Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling" (arXiv 2023)

## License

MIT
