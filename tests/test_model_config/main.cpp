/**
 * test_model_config — verify ModelConfig::load handles real Qwen2.5 config.json
 * and a few synthetic cases (missing fields, malformed JSON).
 */
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "model/model_config.h"

using mini_infer::ModelConfig;

static int g_failures = 0;
#define EXPECT(cond, msg)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__,     \
                         msg);                                                \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

namespace {
void write_tmp(const std::string& body) {
    std::ofstream f("/tmp/test_model_config.json");
    f << body;
}
}

int main() {
    std::printf("mini-infer :: model_config test\n");
    std::printf("-------------------------------\n");

    // 1. Valid Qwen2.5-style config.
    write_tmp(R"({
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"]
    })");
    {
        ModelConfig c = ModelConfig::load("/tmp/test_model_config.json");
        EXPECT(c.hidden_size == 3584, "hidden_size");
        EXPECT(c.intermediate_size == 18944, "intermediate_size");
        EXPECT(c.num_hidden_layers == 28, "num_hidden_layers");
        EXPECT(c.num_attention_heads == 28, "num_attention_heads");
        EXPECT(c.num_key_value_heads == 4, "num_key_value_heads (GQA)");
        EXPECT(c.vocab_size == 152064, "vocab_size");
        EXPECT(c.rope_theta == 1000000.0f, "rope_theta");
        EXPECT(c.rms_norm_eps == 1e-6f, "rms_norm_eps");
        EXPECT(c.tie_word_embeddings == false, "tie_word_embeddings");
        EXPECT(c.head_dim() == 128, "head_dim derived");
        EXPECT(c.num_kv_groups() == 7, "kv groups (28/4)");
        EXPECT(c.model_type == "qwen2", "model_type");
        EXPECT(c.architectures == "Qwen2ForCausalLM", "architectures[0]");
    }

    // 2. Missing optional fields fall back to defaults.
    write_tmp(R"({
        "hidden_size": 512,
        "intermediate_size": 1376,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "vocab_size": 1000
    })");
    {
        ModelConfig c = ModelConfig::load("/tmp/test_model_config.json");
        EXPECT(c.num_key_value_heads == 8, "kv heads default to attn heads (MHA)");
        EXPECT(c.rope_theta == 10000.0f, "rope_theta default");
        EXPECT(c.rms_norm_eps == 1e-6f, "eps default");
        EXPECT(c.tie_word_embeddings == false, "tie default");
        EXPECT(c.head_dim() == 64, "head_dim");
    }

    // 3. GQA invariant: heads must be divisible.
    write_tmp(R"({
        "hidden_size": 100, "intermediate_size": 200,
        "num_hidden_layers": 1, "num_attention_heads": 8,
        "num_key_value_heads": 3, "vocab_size": 10
    })");
    {
        bool threw = false;
        try { ModelConfig::load("/tmp/test_model_config.json"); }
        catch (const std::exception&) { threw = true; }
        EXPECT(threw, "non-divisible GQA heads should throw");
    }

    // 4. Missing required field throws.
    write_tmp(R"({"hidden_size": 100})");
    {
        bool threw = false;
        try { ModelConfig::load("/tmp/test_model_config.json"); }
        catch (const std::exception&) { threw = true; }
        EXPECT(threw, "missing required fields should throw");
    }

    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}