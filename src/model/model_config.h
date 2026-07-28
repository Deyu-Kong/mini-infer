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
 *   - Gemma    : Gemma 1/2/3.  Merged `qkv_proj.weight`, RMSNorm uses
 *                `(1 + weight)` scaling, embeddings scaled by sqrt(hidden).
 */
enum class ModelArch {
    QwenLLaMA = 0,   // Qwen2/2.5 + LLaMA-family decoders (Mistral, Yi, etc.)
    Gemma,           // Gemma 1/2/3
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

    // Mistral: sliding-window attention size. We ignore it for now and use
    // full attention; kept here so the loader is aware.
    int64_t   sliding_window      = 0;

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

    // -------- I/O --------
    static ModelConfig load(const std::string& config_json_path);

    // Classify a `model_type` string into a ModelArch. Pure helper for tests.
    static ModelArch arch_from_model_type(const std::string& model_type);
};

}  // namespace mini_infer