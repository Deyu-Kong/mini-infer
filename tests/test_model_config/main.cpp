/**
 * test_model_config — verify ModelConfig::load handles real Qwen2.5 config.json
 * and a few synthetic cases (missing fields, malformed JSON). Also covers the
 * extended parser that detects LLaMA-family and Gemma configs.
 */
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "model/model_config.h"

using mini_infer::ModelConfig;
using mini_infer::ModelArch;

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
        EXPECT(c.arch == ModelArch::QwenLLaMA, "arch is QwenLLaMA for qwen2");
        EXPECT(c.has_qkv_bias == true, "Qwen2 has QKV bias");
        EXPECT(c.rmsnorm_add_one == false, "Qwen2 doesn't use add_one");
        EXPECT(c.embed_scale == false, "Qwen2 doesn't scale embeddings");
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

    // ---- LLaMA family ---------------------------------------------------

    // 5. LLaMA-3.2-1B style config (MHA + no bias).
    write_tmp(R"({
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "tie_word_embeddings": true,
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    })");
    {
        ModelConfig c = ModelConfig::load("/tmp/test_model_config.json");
        EXPECT(c.arch == ModelArch::QwenLLaMA, "LLaMA arch is QwenLLaMA bucket");
        EXPECT(c.has_qkv_bias == false, "LLaMA has no QKV bias");
        EXPECT(c.rmsnorm_add_one == false, "LLaMA uses standard RMSNorm");
        EXPECT(c.embed_scale == false, "LLaMA doesn't scale embeddings");
        EXPECT(c.tie_word_embeddings == true, "LLaMA-3 ties weights");
        EXPECT(c.rope_theta == 500000.0f, "LLaMA-3 rope_theta");
        EXPECT(c.rms_norm_eps == 1e-5f, "LLaMA-3 rms_norm_eps");
        EXPECT(c.head_dim() == 64, "LLaMA head_dim = 2048/32");
        EXPECT(c.num_kv_groups() == 4, "LLaMA kv groups = 32/8");
    }

    // 6. Mistral: MHA + sliding_window (ignored).
    write_tmp(R"({
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "sliding_window": 4096,
        "model_type": "mistral",
        "architectures": ["MistralForCausalLM"]
    })");
    {
        ModelConfig c = ModelConfig::load("/tmp/test_model_config.json");
        EXPECT(c.arch == ModelArch::QwenLLaMA, "Mistral is QwenLLaMA bucket");
        EXPECT(c.has_qkv_bias == false, "Mistral has no QKV bias");
        EXPECT(c.sliding_window == 4096, "Mistral sliding_window captured");
        EXPECT(c.head_dim() == 128, "Mistral head_dim = 4096/32");
    }

    // ---- Gemma ----------------------------------------------------------

    // 7. Gemma-2 style config (head_dim explicit, add_one, embed scale).
    write_tmp(R"({
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 256000,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "model_type": "gemma2",
        "architectures": ["Gemma2ForCausalLM"]
    })");
    {
        ModelConfig c = ModelConfig::load("/tmp/test_model_config.json");
        EXPECT(c.arch == ModelArch::Gemma, "gemma2 -> Gemma arch");
        EXPECT(c.rmsnorm_add_one == true, "Gemma uses (1 + weight) RMSNorm");
        EXPECT(c.embed_scale == true, "Gemma scales embeddings");
        EXPECT(c.has_qkv_bias == false, "Gemma has no QKV bias");
        EXPECT(c.head_dim() == 256, "Gemma explicit head_dim wins over hidden/heads");
        EXPECT(c.tie_word_embeddings == true, "Gemma2 ties weights");
        EXPECT(c.num_kv_groups() == 2, "Gemma2 kv groups = 8/4");
    }

    // 8. Gemma-1 (model_type = "gemma").
    write_tmp(R"({
        "hidden_size": 2048,
        "intermediate_size": 16384,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "vocab_size": 256000,
        "rope_theta": 10000.0,
        "model_type": "gemma",
        "architectures": ["GemmaForCausalLM"]
    })");
    {
        ModelConfig c = ModelConfig::load("/tmp/test_model_config.json");
        EXPECT(c.arch == ModelArch::Gemma, "gemma -> Gemma arch");
        EXPECT(c.rmsnorm_add_one == true, "Gemma uses add_one RMSNorm");
        EXPECT(c.embed_scale == true, "Gemma scales embeddings");
        EXPECT(c.head_dim() == 256, "Gemma-1 head_dim explicit");
    }

    // 9. arch_from_model_type pure helper.
    EXPECT(ModelConfig::arch_from_model_type("qwen2")   == ModelArch::QwenLLaMA,
           "qwen2 -> QwenLLaMA");
    EXPECT(ModelConfig::arch_from_model_type("llama")   == ModelArch::QwenLLaMA,
           "llama -> QwenLLaMA");
    EXPECT(ModelConfig::arch_from_model_type("mistral") == ModelArch::QwenLLaMA,
           "mistral -> QwenLLaMA");
    EXPECT(ModelConfig::arch_from_model_type("yi")      == ModelArch::QwenLLaMA,
           "yi -> QwenLLaMA");
    EXPECT(ModelConfig::arch_from_model_type("deepseek")== ModelArch::QwenLLaMA,
           "deepseek -> QwenLLaMA");
    EXPECT(ModelConfig::arch_from_model_type("gemma")   == ModelArch::Gemma,
           "gemma -> Gemma");
    EXPECT(ModelConfig::arch_from_model_type("gemma2")  == ModelArch::Gemma,
           "gemma2 -> Gemma");
    EXPECT(ModelConfig::arch_from_model_type("gemma3_text") == ModelArch::Gemma,
           "gemma3_text -> Gemma");

    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}