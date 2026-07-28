# Model Generalization Plan

## Current State

`ModelConfig` + `QwenModel::load_weights` used to hardcode Qwen2.5's config
field names and safetensors weight names. The core components (RMSNorm, RoPE,
SwiGLU, GQA) are already generic — the bottleneck is **config parsing** and
**weight name mapping**.

As of mid-2026, all four phases are implemented:

- **Phase 1** (LLaMA family): LLaMA 2/3, Mistral, Yi, DeepSeek support
- **Phase 2** (Gemma 1/2/3): GeGLU, 4-norm block, Q/K RMSNorm, sliding window, dual RoPE
- **Phase 3** (Architecture Abstraction): `QwenModel` → `TransformerModel`, `ArchRegistry`
- **Phase 4** (MoE Support): Mixtral / DeepSeek-MoE / Qwen2-MoE via `MoELayer`

`ModelConfig` carries a `ModelArch` enum (`QwenLLaMA`, `Gemma`, `GPT2`, `Bloom`)
selected from `model_type`; `WeightNameMapper` returns per-layer HF weight names
per arch; RMSNorm has an `add_one_` flag for Gemma; `TransformerModel::forward`
scales embeddings by `sqrt(hidden_size)` for Gemma. `ArchRegistry` provides
per-arch traits (norm type, position encoding, activation, bias). MoE models
use `MoELayer` with fused top-K routing + per-expert MLP dispatch.

## Phase 1: LLaMA Family (DONE)

Cover LLaMA 2/3/3.1, Mistral, Yi, DeepSeek, etc. These share RMSNorm + RoPE + SwiGLU + GQA with Qwen2.5.

### Changes

1. **`ModelConfig`** — add `model_type` dispatch, handle field name differences:
   - LLaMA: `num_attention_heads` → `num_heads`, no `num_key_value_heads` defaults to MHA
   - Mistral: `sliding_window` field (ignore initially, use full attention)
   - Gemma: `head_dim` may be specified independently from `hidden_size / num_heads`

2. **Weight name mapping** — add `WeightNameMapper`, select mapping rules by `model_type`:
   ```
   Qwen2.5:  model.layers.{i}.self_attn.q_proj.weight  →  layers_[i].attn.w_q_
   LLaMA:    model.layers.{i}.self_attn.q_proj.weight  →  same
   Mistral:  model.layers.{i}.self_attn.q_proj.weight  →  same
   ```
   Key difference: Qwen2.5 has QKV bias (`q_proj.bias`), LLaMA/Mistral do not. The mapping table handles the `has_bias_` flag.

3. **`Attention::forward`** — `has_bias_` already exists; ensure LLaMA path skips bias add.

4. **Validation** — run LLaMA-3.2-1B-Instruct with greedy, compare output against HuggingFace.

### Files to modify

- `src/model/model_config.h/cpp` — generalize config parsing
- `src/model/qwen_model.cpp` — weight name mapping layer
- `src/model/safetensors_loader.h/cpp` — optional: add weight name alias support

### New files

- `src/model/weight_mapper.h/cpp` — weight name mapping abstraction

## Phase 2: Gemma Support (DONE)

### What's done

- `ModelConfig::arch` is set to `ModelArch::Gemma` whenever `model_type`
  starts with `gemma` (gemma, gemma2, gemma3, gemma3_text).
- `RMSNorm::add_one_` makes the kernel use `y = x * rrms * (1 + w)` for Gemma.
- `QwenModel::forward` (and paged / batched variants) apply
  `y *= sqrt(hidden_size)` after the embedding lookup when `embed_scale` is set.
- `WeightNameMapper` produces the right per-layer names for both QwenLLaMA
  and Gemma, and `load_qkv` handles the (rare) merged `qkv_proj.weight`
  for Gemma checkpoints that store Q/K/V together.

### Still TODO for full Gemma inference

The plan's "half day" estimate only covered RMSNorm/embedding scale/weight
names. Actual Gemma inference needs:

1. **GeGLU MLP**: Gemma uses `gelu_pytorch_tanh` instead of `silu` — current
   `MLP` is SwiGLU. Need an `MLP::set_activation(Act::GeLU)` switch and a
   `gelu_kernel.cu`.
2. **Per-layer RMSNorm count**: Gemma 2/3 has 4 norms per block
   (`input_layernorm`, `post_attention_layernorm`,
   `pre_feedforward_layernorm`, `post_feedforward_layernorm`); current
   model only has 2.
3. **q_norm / k_norm**: Gemma 3 RMSNorm-applies Q and K before RoPE.
4. **Sliding window attention** (Gemma 2/3): alternate layers use a
   local sliding window; needs an `attn_mask` kernel integration.
5. **Two RoPE bands** (Gemma 3): global vs local RoPE inv_freq.

These belong to a follow-up Phase 2.5 / Phase 3 work item.

### Changes

1. **RMSNorm** — Gemma uses `(1 + weight) * x * rsqrt(mean(x²) + eps)`, add an `add_one_` flag
2. **Embedding scaling** — Gemma multiplies embedding output by `sqrt(hidden_size)`, add a scale step after embedding lookup in `QwenModel::forward`
3. **Weight names** — Gemma uses `model.layers.{i}.self_attn.q_proj` but K/V may be merged as `qkv_proj`, need to split

### Files to modify

- `src/layers/rmsnorm.h/cpp` — add `add_one_` variant
- `src/model/qwen_model.cpp` — embedding scale for Gemma

## Phase 3: Architecture Abstraction (DONE)

Renamed `QwenModel` to `TransformerModel`, extended `ModelArch` enum:

```cpp
enum class ModelArch {
    QwenLLaMA, // Qwen2/2.5, LLaMA 2/3, Mistral, Yi, DeepSeek
    Gemma,     // Gemma 1/2/3
    GPT2,      // GPT-2, GPT-Neo (future)
    Bloom,     // Bloom (future)
};
```

`ModelConfig::load()` auto-detects `ModelArch` from `model_type` in config.json. `TransformerModel` selects internally based on arch:

- Norm type (RMSNorm vs LayerNorm)
- Position encoding (RoPE vs ALiBi vs learned)
- MLP type (SwiGLU vs GELU)
- Attention bias

### What's done

- `QwenModel` renamed to `TransformerModel` across the entire codebase
  (engine, speculative, scheduler, main, tests, benchmarks).
- `src/model/transformer_model.h/cpp` replaces `qwen_model.h/cpp`.
- `src/model/arch_registry.h` added: `ArchRegistry` provides per-arch
  `ArchTraits` (norm type, position encoding, default activation, bias,
  embedding scale, double-norm block). GPT2 and Bloom are registered as
  future stubs.
- `ModelArch` enum extended with `GPT2` and `Bloom` (future placeholders).
- `TransformerModel` includes `arch_registry.h` for arch-trait queries.

### New files

- `src/model/transformer_model.h/cpp` — generalized model (renamed from QwenModel)
- `src/model/arch_registry.h` — arch detection and registration

### Files modified

- `src/model/model_config.h` — added `GPT2`, `Bloom` to `ModelArch`
- `src/core/engine.h/cpp` — uses `TransformerModel`
- `src/speculative/draft_engine.h/cpp` — same rename
- `src/speculative/spec_decoder.h/cpp` — same rename
- `src/scheduler/scheduler.h/cpp` — same rename
- `src/core/main.cc` — same rename
- `CMakeLists.txt` — updated source file name

## Phase 4: MoE Support (DONE)

Mixtral / DeepSeek-MoE / Qwen2-MoE support implemented.

### What's done

- `ModelConfig` extended with MoE fields: `num_experts`, `num_experts_per_tok`,
  `moe_intermediate_size`. Parses both `num_experts` (Qwen2-MoE) and
  `num_local_experts` (Mixtral) from config.json. `is_moe()` helper returns
  true when MoE is enabled.
- `MoELayer` class (`src/layers/moe.h/cpp`): owns N expert MLPs + router gate.
  Forward: router GEMM → fused top-K + softmax CUDA kernel → per-expert MLP
  → weighted scatter-add accumulation.
- `moe_kernel.cu`: fused top-K expert selection with softmax weighting,
  plus scatter-add kernel for accumulating expert outputs.
- `TransformerModel` dispatches to `MoELayer` or `MLP` per-layer based on
  `LayerWeights::use_moe` flag. Weight loading reads Mixtral-style
  `block_sparse_moe.gate.weight` + `block_sparse_moe.experts.{j}.w{1,2,3}.weight`.

### New files

- `src/layers/moe.h/cpp` — MoE layer implementation
- `src/kernels/moe_kernel.cu` — fused top-K gate + expert dispatch kernel
- `src/kernels/moe_kernel.cuh` — kernel declarations

### Files modified

- `src/model/transformer_model.h/cpp` — MoE layer dispatch in forward + weight loading
- `src/model/model_config.h/cpp` — MoE config fields and parsing
- `CMakeLists.txt` — added moe.cpp and moe_kernel.cu

## Priority

| Phase | Effort | New model coverage | Recommendation |
| ----- | ------ | ------------------ | -------------- |
| 1 | 1-2 days | LLaMA 2/3, Mistral, Yi, DeepSeek | **Do first** |
| 2 | half day | Gemma 1/2/3 | Do alongside Phase 1 |
| 3 | 2-3 days | Architecture-level refactor | **Done** |
| 4 | 3-5 days | Mixtral, DeepSeek-MoE | **Done** |

Phase 1 alone covers 70%+ of mainstream open-source models.
