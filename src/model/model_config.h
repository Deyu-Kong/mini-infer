#pragma once

#include <cstdint>
#include <string>

namespace mini_infer {

/**
 * ModelConfig — holds the architectural parameters parsed from a HuggingFace
 * `config.json`. Currently covers the Qwen2 / LLaMA-style decoder used by
 * Qwen2.5-7B-Instruct and friends.
 *
 * Fields are populated by `ModelConfig::load(path)`; missing optional keys
 * fall back to defaults so the loader is tolerant of config drift.
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
    float   rope_theta             = 1000000.0f;
    int64_t max_position_embeddings = 32768;

    // -------- optional / housekeeping --------
    bool    tie_word_embeddings    = false;
    std::string hidden_act          = "silu";
    std::string torch_dtype         = "bfloat16";   // what the safetensors store
    std::string model_type          = "qwen2";
    std::string architectures       = "Qwen2ForCausalLM";

    // -------- derived --------
    int64_t head_dim() const {
        return hidden_size / num_attention_heads;
    }
    int64_t kv_head_dim() const {
        // GQA: kv head dim is the same as attn head dim (projected separately)
        return head_dim();
    }
    int64_t num_kv_groups() const {
        return num_attention_heads / num_key_value_heads;
    }

    // -------- I/O --------
    static ModelConfig load(const std::string& config_json_path);
};

}  // namespace mini_infer