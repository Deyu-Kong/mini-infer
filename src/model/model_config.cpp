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
    c.hidden_act              = jget_str (j, "hidden_act", "silu");
    c.torch_dtype             = jget_str (j, "torch_dtype", "bfloat16");
    c.model_type              = jget_str (j, "model_type", "qwen2");
    if (j.contains("architectures") && j["architectures"].is_array() &&
        !j["architectures"].empty()) {
        c.architectures = j["architectures"][0].get<std::string>();
    }

    if (c.hidden_size == 0 || c.num_hidden_layers == 0 ||
        c.num_attention_heads == 0 || c.vocab_size == 0) {
        throw std::runtime_error("ModelConfig: missing required fields in " + path);
    }
    if (c.hidden_size % c.num_attention_heads != 0) {
        throw std::runtime_error("ModelConfig: hidden_size not divisible by num_attention_heads");
    }
    if (c.num_attention_heads % c.num_key_value_heads != 0) {
        throw std::runtime_error("ModelConfig: num_attention_heads not divisible by num_key_value_heads (GQA)");
    }
    return c;
}

}  // namespace mini_infer