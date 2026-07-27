# Model Generalization Plan

## Current State

`ModelConfig` + `QwenModel::load_weights` hardcode Qwen2.5's config field names and safetensors weight names. The core components (RMSNorm, RoPE, SwiGLU, GQA) are already generic — the bottleneck is **config parsing** and **weight name mapping**.

## Phase 1: LLaMA Family (1-2 days)

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

## Phase 2: Gemma Support (half day)

### Changes

1. **RMSNorm** — Gemma uses `(1 + weight) * x * rsqrt(mean(x²) + eps)`, add an `add_one_` flag
2. **Embedding scaling** — Gemma multiplies embedding output by `sqrt(hidden_size)`, add a scale step after embedding lookup in `QwenModel::forward`
3. **Weight names** — Gemma uses `model.layers.{i}.self_attn.q_proj` but K/V may be merged as `qkv_proj`, need to split

### Files to modify

- `src/layers/rmsnorm.h/cpp` — add `add_one_` variant
- `src/model/qwen_model.cpp` — embedding scale for Gemma

## Phase 3: Architecture Abstraction (2-3 days, optional)

Rename `QwenModel` to `TransformerModel`, introduce `ModelArch` enum:

```cpp
enum class ModelArch {
    LLaMA,    // Qwen2.5, LLaMA 2/3, Mistral, Yi, DeepSeek
    Gemma,    // Gemma 1/2/3
    GPT2,     // GPT-2, GPT-Neo (future)
    Bloom,    // Bloom (future)
};
```

`ModelConfig::load()` auto-detects `ModelArch` from `model_type` in config.json. `TransformerModel` selects internally based on arch:

- Norm type (RMSNorm vs LayerNorm)
- Position encoding (RoPE vs ALiBi vs learned)
- MLP type (SwiGLU vs GELU)
- Attention bias

### New files

- `src/model/transformer_model.h/cpp` — generalized model (rename from QwenModel)
- `src/model/arch_registry.h` — arch detection and registration

### Files to modify

- `src/model/model_config.h/cpp` — add `ModelArch` enum and detection
- `src/core/engine.h/cpp` — use `TransformerModel` instead of `QwenModel`
- `src/speculative/draft_engine.h/cpp` — same rename
- `src/core/main.cc` — same rename

## Phase 4: MoE Support (3-5 days, optional)

Mixtral / DeepSeek-MoE / Qwen2-MoE require:

- New `MoELayer` class: router gate + top-K expert selection + weighted sum
- `TransformerModel` selects dense MLP or MoE based on arch

### New files

- `src/layers/moe.h/cpp` — MoE layer implementation
- `src/kernels/moe_kernel.cu` — fused top-K gate + expert dispatch kernel

### Files to modify

- `src/model/transformer_model.cpp` — MoE layer dispatch
- `src/model/model_config.h` — MoE config fields (`num_experts`, `num_experts_per_tok`)

## Priority

| Phase | Effort | New model coverage | Recommendation |
| ----- | ------ | ------------------ | -------------- |
| 1 | 1-2 days | LLaMA 2/3, Mistral, Yi, DeepSeek | **Do first** |
| 2 | half day | Gemma 1/2/3 | Do alongside Phase 1 |
| 3 | 2-3 days | Architecture-level refactor | When time permits |
| 4 | 3-5 days | Mixtral, DeepSeek-MoE | Nice to have |

Phase 1 alone covers 70%+ of mainstream open-source models.
