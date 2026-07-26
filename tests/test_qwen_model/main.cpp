/**
 * test_qwen_model — exercise QwenModel::load_weights on the real
 * Qwen2.5-Coder-7B-Instruct copy (acts as Qwen2.5-7B-Instruct since the
 * architecture is identical). Skipped if the model dir is absent.
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

#include "model/model_config.h"
#include "model/qwen_model.h"
#include "model/safetensors_loader.h"

using mini_infer::ModelConfig;
using mini_infer::QwenModel;
using mini_infer::WeightIndex;

static int g_failures = 0;
#define EXPECT(cond, msg)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__,     \
                         msg);                                                \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

static const char* kModelDir = "/data1/kdy/LLMs/Qwen2.5-Coder-7B-Instruct";

int main() {
    std::printf("mini-infer :: qwen_model test\n");
    std::printf("-----------------------------\n");

    if (::access(kModelDir, R_OK) != 0) {
        std::printf("SKIP: model dir not found at %s\n", kModelDir);
        return 0;
    }

    // 1. Parse config.
    ModelConfig cfg = ModelConfig::load(std::string(kModelDir) + "/config.json");
    std::printf("  config: H=%ld I=%ld L=%ld Hq=%ld Hkv=%ld V=%ld\n",
                cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers,
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.num_key_value_heads * cfg.kv_head_dim(),
                cfg.vocab_size);
    EXPECT(cfg.hidden_size == 3584, "hidden_size 3584");
    EXPECT(cfg.num_hidden_layers == 28, "28 layers");
    EXPECT(cfg.num_key_value_heads == 4, "GQA: 4 kv heads");

    // 2. Build QwenModel and load weights. This copies ~15 GB BF16 -> FP16
    //    from the mmap'd shards onto the GPU; takes ~1-2 s on A6000.
    QwenModel model(cfg, /*device=*/0);
    WeightIndex idx = WeightIndex::load(kModelDir);
    model.load_weights(idx);
    std::printf("  load_weights complete\n");

    // 3. Spot-check a couple of weight pointers.
    EXPECT(model.embed_tokens().numel() == cfg.vocab_size * cfg.hidden_size,
           "embed shape");
    EXPECT(model.lm_head().numel() == cfg.vocab_size * cfg.hidden_size,
           "lm_head shape");
    EXPECT(model.layers().size() == static_cast<size_t>(cfg.num_hidden_layers),
           "28 layers instantiated");

    cudaDeviceSynchronize();
    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}