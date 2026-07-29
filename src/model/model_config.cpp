#include "model/model_config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace mini_infer {

namespace {

// Read an integer-like field, falling back to default if missing.
int64_t jget_int(const nlohmann::json& j, const char* key, int64_t def) {
    if (!j.contains(key)) return def;
    if (j[key].is_number_integer()) return j[key].get<int64_t>();
    if (j[key].is_number_float())   return static_cast<int64_t>(j[key].get<double>());
    if (j[key].is_string())         return std::stoll(j[key].get<std::string>());
    throw std::runtime_error(std::string("ModelConfig: bad integer field ") + key);
}

float jget_float(const nlohmann::json& j, const char* key, float def) {
    if (!j.contains(key)) return def;
    if (j[key].is_number()) return j[key].get<float>();
    if (j[key].is_string()) return std::stof(j[key].get<std::string>());
    throw std::runtime_error(std::string("ModelConfig: bad float field ") + key);
}

bool jget_bool(const nlohmann::json& j, const char* key, bool def) {
    if (!j.contains(key)) return def;
    if (j[key].is_boolean()) return j[key].get<bool>();
    throw std::runtime_error(std::string("ModelConfig: bad bool field ") + key);
}

std::string jget_str(const nlohmann::json& j, const char* key,
                     const std::string& def) {
    if (!j.contains(key)) return def;
    return j[key].get<std::string>();
}

}  // namespace

ModelArch ModelConfig::arch_from_model_type(const std::string& mt) {
    // Gemma uses model_type "gemma", "gemma2", "gemma3", "gemma3_text".
    if (mt.rfind("gemma", 0) == 0) return ModelArch::Gemma;
    // Everything else currently falls into the Qwen / LLaMA bucket. This
    // includes: qwen2, qwen2_vl, llama, mistral, mixtral, yi, deepseek,
    // phi3 (architecturally LLaMA-ish), gemma2 (handled above), etc.
    return ModelArch::QwenLLaMA;
}

ActKind ModelConfig::act_from_string(const std::string& ha) {
    // PyTorch names. Gemma uses "gelu_pytorch_tanh".
    if (ha == "silu" || ha == "SwiGLU") return ActKind::Silu;
    if (ha == "gelu_pytorch_tanh" || ha == "gelu") return ActKind::GeluTanh;
    // Unknown default to SwiGLU (matches Qwen/LLaMA/Mistral/Yi/DeepSeek).
    return ActKind::Silu;
}

ModelConfig ModelConfig::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("ModelConfig: cannot open " + path);
    nlohmann::json j;
    f >> j;

    ModelConfig c;
    c.hidden_size             = jget_int(j, "hidden_size", 0);
    c.intermediate_size        = jget_int(j, "intermediate_size", 0);
    c.num_hidden_layers       = jget_int(j, "num_hidden_layers", 0);
    c.num_attention_heads     = jget_int(j, "num_attention_heads", 0);
    c.num_key_value_heads     = jget_int(j, "num_key_value_heads",
                                          c.num_attention_heads);  // MHA fallback
    c.vocab_size              = jget_int(j, "vocab_size", 0);
    c.rms_norm_eps            = jget_float(j, "rms_norm_eps", 1e-6f);
    c.rope_theta              = jget_float(j, "rope_theta", 10000.0f);
    c.max_position_embeddings = jget_int(j, "max_position_embeddings", 32768);
    c.tie_word_embeddings     = jget_bool(j, "tie_word_embeddings", false);
    // Some Gemma configs use `hidden_activation` instead of `hidden_act`.
    if (j.contains("hidden_act")) {
        c.hidden_act = j["hidden_act"].get<std::string>();
    } else if (j.contains("hidden_activation")) {
        c.hidden_act = j["hidden_activation"].get<std::string>();
    }
    c.torch_dtype             = jget_str (j, "torch_dtype", "bfloat16");
    c.model_type              = jget_str (j, "model_type", "qwen2");
    if (j.contains("architectures") && j["architectures"].is_array() &&
        !j["architectures"].empty()) {
        c.architectures = j["architectures"][0].get<std::string>();
    }

    c.mlp_act = act_from_string(c.hidden_act);

    // ---- MoE fields ----------------------------------------------------
    c.num_experts = jget_int(j, "num_experts", 0);
    c.num_experts_per_tok = jget_int(j, "num_experts_per_tok", 0);
    c.moe_intermediate_size = jget_int(j, "moe_intermediate_size", 0);
    // Mixtral uses "num_local_experts"
    if (c.num_experts == 0) {
        c.num_experts = jget_int(j, "num_local_experts", 0);
    }
    // DeepSeek uses "n_routed_experts"
    if (c.num_experts == 0) {
        c.num_experts = jget_int(j, "n_routed_experts", 0);
    }
    // Fallback: if MoE is enabled but moe_intermediate_size is not set,
    // use the dense intermediate_size (some configs do this).
    if (c.num_experts > 0 && c.moe_intermediate_size == 0) {
        c.moe_intermediate_size = c.intermediate_size;
    }
    // Qwen2-MoE shared expert intermediate size
    c.shared_expert_intermediate_size = jget_int(j, "shared_expert_intermediate_size", 0);

    // ---- arch-specific overrides --------------------------------------
    c.arch = arch_from_model_type(c.model_type);
    if (c.arch == ModelArch::Gemma) {
        c.head_dim_override = jget_int(j, "head_dim", 0);
        c.rmsnorm_add_one = true;
        c.embed_scale = true;
        c.has_qkv_bias = false;
        if (!j.contains("tie_word_embeddings")) {
            c.tie_word_embeddings = true;
        }
        if (c.rope_theta == 0.0f) c.rope_theta = 10000.0f;

        // Gemma 2 (model_type == "gemma2") and Gemma 3 use the 4-norm
        // double-wrapped block. Gemma 1 (model_type == "gemma") uses
        // the LLaMA-style 2-norm block but with GeGLU.
        c.double_norm_block = (c.model_type == "gemma2" ||
                               c.model_type.rfind("gemma3", 0) == 0);

    // Sliding window attention. Gemma 2 and 3 both use it; Gemma 1
    // does not.
    c.sliding_window = jget_int(j, "sliding_window", 0);
    if (c.model_type == "gemma2") {
        // Gemma 2: alternating pattern (even layers = sliding, odd = full)
        c.sliding_window_pattern = 2;
    } else if (c.model_type.rfind("gemma3", 0) == 0) {
        c.sliding_window_pattern = jget_int(j, "sliding_window_pattern", 6);
        c.use_qk_norm = true;
        c.dual_rope = true;
        c.local_rope_theta = jget_float(j, "rope_local_base_freq", 10000.0f);
    }
    } else {
        // Qwen2 / LLaMA family.
        const std::string& a = c.architectures;
        if (a.rfind("Qwen", 0) == 0 || a.rfind("qwen", 0) == 0) {
            c.has_qkv_bias = true;
        } else {
            c.has_qkv_bias = false;
        }
        // Mistral sliding window (ignored for now; we run full attention).
        c.sliding_window = jget_int(j, "sliding_window", 0);
    }

    // ---- validation ----------------------------------------------------
    if (c.hidden_size == 0 || c.num_hidden_layers == 0 ||
        c.num_attention_heads == 0 || c.vocab_size == 0) {
        throw std::runtime_error("ModelConfig: missing required fields in " + path);
    }
    const int64_t hd = c.head_dim();
    if (hd <= 0) {
        throw std::runtime_error(
            "ModelConfig: head_dim is non-positive (hidden_size=" +
            std::to_string(c.hidden_size) + ", num_attention_heads=" +
            std::to_string(c.num_attention_heads) + ", head_dim_override=" +
            std::to_string(c.head_dim_override) + ")");
    }
    if (c.hidden_size != hd * c.num_attention_heads && c.head_dim_override == 0) {
        throw std::runtime_error(
            "ModelConfig: hidden_size not divisible by num_attention_heads");
    }
    if (c.num_attention_heads % c.num_key_value_heads != 0) {
        throw std::runtime_error(
            "ModelConfig: num_attention_heads not divisible by num_key_value_heads (GQA)");
    }
    return c;
}

}  // namespace mini_infer