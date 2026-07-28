#pragma once

#include <cstdint>
#include <string>

namespace mini_infer {

/**
 * Architecture family detected from a model's `config.json`. Drives
 * weight-name mapping, norm formulation, and other per-family quirks
 * in the model loader.
 *
 *   - QwenLLaMA: Qwen2/2.5, LLaMA 2/3/3.1, Mistral, Yi, DeepSeek.
 *                Same HF safetensors naming; q_proj/k_proj/v_proj split.
 *                Qwen models have QKV bias; LLaMA family does not.
 *   - Gemma    : Gemma 1/2/3.  RMSNorm uses `(1 + weight)` scaling,
 *                embeddings scaled by sqrt(hidden). Uses GeGLU MLP.
 *                Sub-arch flags below distinguish Gemma 1 (LLaMA-style
 *                block, no extra norms) from Gemma 2/3 (4-norm block
 *                with double-norm wrapping, sliding window).
 */
enum class ModelArch {
    QwenLLaMA = 0,   // Qwen2/2.5 + LLaMA-family decoders (Mistral, Yi, etc.)
    Gemma,           // Gemma 1/2/3
    GPT2,            // GPT-2, GPT-Neo (future)
    Bloom,           // Bloom (future)
};

/**
 * Activation function used by the MLP gate.
 *   - Silu     : silu(x) = x * sigmoid(x). Used by SwiGLU (Qwen/LLaMA/Mistral).
 *   - GeluTanh : PyTorch `gelu_pytorch_tanh` — GELU with the tanh
 *                approximation. Used by GeGLU (Gemma 1/2/3).
 */
enum class ActKind {
    Silu = 0,
    GeluTanh,
};

/**
 * ModelConfig — holds the architectural parameters parsed from a HuggingFace
 * `config.json`. Covers Qwen2/2.5, LLaMA 2/3/3.1, Mistral, Yi, DeepSeek, Gemma.
 *
 * Fields are populated by `ModelConfig::load(path)`. Missing optional keys
 * fall back to defaults so the loader is tolerant of config drift.
 *
 * `model_type` is the value of the `model_type` field in config.json
 * (e.g. "qwen2", "llama", "mistral", "gemma2"). It is used both as an
 * informational string and to drive arch detection.
 */
struct ModelConfig {
    // -------- required --------
    int64_t hidden_size            = 0;
    int64_t intermediate_size       = 0;
    int64_t num_hidden_layers      = 0;
    int64_t num_attention_heads    = 0;
    int64_t num_key_value_heads    = 0;       // GQA: typically < num_attention_heads
    int64_t vocab_size             = 0;
    float   rms_norm_eps           = 1e-6f;
    float   rope_theta             = 10000.0f;
    int64_t max_position_embeddings = 32768;

    // -------- optional / housekeeping --------
    bool    tie_word_embeddings    = false;
    std::string hidden_act          = "silu";
    std::string torch_dtype         = "bfloat16";   // what the safetensors store
    std::string model_type          = "qwen2";
    std::string architectures       = "Qwen2ForCausalLM";

    // -------- arch-specific (set after load()) --------
    ModelArch arch                = ModelArch::QwenLLaMA;
    bool      has_qkv_bias        = false;     // Qwen2 has it; LLaMA family does not
    bool      rmsnorm_add_one     = false;     // Gemma uses (1 + weight) * x
    bool      embed_scale         = false;     // Gemma scales embeds by sqrt(hidden)
    int64_t   head_dim_override   = 0;         // Gemma sets head_dim explicitly
    ActKind   mlp_act             = ActKind::Silu;  // SwiGLU (default) vs GeGLU (Gemma)

    // Mistral / Gemma: sliding-window attention size. 0 disables.
    int64_t   sliding_window      = 0;
    // Gemma 3: every `sliding_window_pattern`-th layer (0-indexed) uses
    // the local (sliding) attention. E.g. pattern=6 means layers
    // 5, 11, 17, ... are sliding, the rest are global.
    int64_t   sliding_window_pattern = 0;

    // Gemma 2/3 use a 4-norm block (double-norm wrapping) instead of the
    // standard LLaMA-style 2-norm block.
    bool      double_norm_block   = false;

    // Gemma 3 RMSNorm-applies Q and K before RoPE.
    bool      use_qk_norm         = false;
    // Gemma 3 dual-band RoPE: every odd-indexed layer uses a separate
    // `local_rope_theta` for inv_freq, in addition to `rope_theta` (global).
    bool      dual_rope           = false;
    float     local_rope_theta    = 10000.0f;

    // MoE (Mixture of Experts) — Mixtral, DeepSeek-MoE, Qwen2-MoE.
    // When num_experts > 0, the MLP is replaced by a sparse MoE layer.
    int64_t   num_experts         = 0;          // 0 = dense MLP
    int64_t   num_experts_per_tok = 0;          // top-K experts per token
    int64_t   moe_intermediate_size = 0;        // per-expert intermediate size

    // -------- derived --------
    int64_t head_dim() const {
        if (head_dim_override > 0) return head_dim_override;
        return hidden_size / num_attention_heads;
    }
    int64_t kv_head_dim() const {
        return head_dim();   // GQA: kv head dim is the same as attn head dim
    }
    int64_t num_kv_groups() const {
        return num_attention_heads / num_key_value_heads;
    }

    // True iff this model uses MoE (sparse MLP) layers.
    bool is_moe() const { return num_experts > 0 && num_experts_per_tok > 0; }

    // True iff this layer should run with sliding (local) attention.
    // For LLaMA/Mistral/Qwen/Yi/DeepSeek/Gemma-1/Gemma-2: always false.
    // For Gemma 3 with sliding_window_pattern > 0: every pattern-th
    // layer is sliding, the rest are global.
    bool is_layer_sliding(int64_t layer_idx) const {
        if (sliding_window <= 0 || sliding_window_pattern <= 0) return false;
        // Gemma 2 (pattern=2): even layers (0, 2, 4, ...) are sliding
        if (sliding_window_pattern == 2) {
            return layer_idx % 2 == 0;
        }
        // Gemma 3 (pattern=6): every 6th layer (5, 11, 17, ...) is full attention,
        // others are sliding
        return (layer_idx + 1) % sliding_window_pattern != 0;
    }

    // -------- I/O --------
    static ModelConfig load(const std::string& config_json_path);

    // Classify a `model_type` string into a ModelArch. Pure helper for tests.
    static ModelArch arch_from_model_type(const std::string& model_type);

    // Map a `hidden_act` string to an ActKind.
    static ActKind act_from_string(const std::string& hidden_act);
};

}  // namespace mini_infer