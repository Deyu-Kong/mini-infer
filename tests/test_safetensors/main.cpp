/**
 * test_safetensors — verify SafetensorsLoader and WeightIndex on a real
 * Qwen2.5 model. Skipped (passed with "skipped") if the model directory is
 * not present, so CI without network access stays green.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "model/safetensors_loader.h"

using mini_infer::SafetensorsLoader;
using mini_infer::WeightIndex;
using mini_infer::WeightInfo;

static int g_failures = 0;
#define EXPECT(cond, msg)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__,     \
                         msg);                                                \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

// Default location of the dev Qwen2.5-7B copy.
static const char* kModelDir = "/data1/kdy/LLMs/Qwen2.5-Coder-7B-Instruct";

int main() {
    std::printf("mini-infer :: safetensors test\n");
    std::printf("------------------------------\n");

    // Check the directory exists; otherwise skip gracefully.
    if (::access(kModelDir, R_OK) != 0) {
        std::printf("SKIP: model dir not found at %s\n", kModelDir);
        return 0;
    }

    // 1. Open the WeightIndex (multi-shard via index.json).
    WeightIndex idx;
    try {
        idx = WeightIndex::load(kModelDir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: WeightIndex::load threw: %s\n", e.what());
        return 1;
    }
    std::printf("  loaded %zu tensors from %s\n", idx.num_tensors(), kModelDir);

    // 2. Sanity-check a handful of well-known tensors + their shapes.
    struct Want { const char* name; std::vector<int64_t> shape; };
    const Want wants[] = {
        {"model.embed_tokens.weight",          {152064, 3584}},
        {"model.layers.0.input_layernorm.weight",         {3584}},
        {"model.layers.0.post_attention_layernorm.weight", {3584}},
        {"model.layers.0.self_attn.q_proj.weight", {3584, 3584}},
        {"model.layers.0.self_attn.k_proj.weight", {512,  3584}},  // GQA: Hkv=4*128=512
        {"model.layers.0.self_attn.v_proj.weight", {512,  3584}},
        {"model.layers.0.self_attn.o_proj.weight", {3584, 3584}},
        {"model.layers.0.mlp.gate_proj.weight",    {18944, 3584}},
        {"model.layers.0.mlp.up_proj.weight",      {18944, 3584}},
        {"model.layers.0.mlp.down_proj.weight",    {3584, 18944}},
        {"lm_head.weight",                        {152064, 3584}},
        {"model.norm.weight",                     {3584}},
    };
    auto print_shape = [](const std::vector<int64_t>& s) {
        std::printf("[");
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) std::printf(",");
            std::printf("%ld", s[i]);
        }
        std::printf("]");
    };
    for (const auto& w : wants) {
        const WeightInfo* wi = idx.find(w.name);
        EXPECT(wi != nullptr, ("exists: " + std::string(w.name)).c_str());
        if (!wi) continue;
        EXPECT(wi->shape == w.shape, ("shape: " + std::string(w.name)).c_str());
        EXPECT(wi->byte_size == static_cast<std::size_t>(wi->numel()) *
                                 mini_infer::dtype_size(wi->dtype),
               ("byte_size: " + std::string(w.name)).c_str());
        std::printf("  %-50s shape=", w.name);
        print_shape(wi->shape);
        std::printf(" dtype=%s\n", mini_infer::dtype_name(wi->dtype));
    }

    // 3. Read a small tensor (input_layernorm) and check it round-trips.
    {
        auto t = idx.read_to_cpu("model.layers.0.input_layernorm.weight");
        EXPECT(t.dtype() == mini_infer::DType::BF16, "input_layernorm dtype BF16");
        EXPECT(t.numel() == 3584, "input_layernorm numel");
        EXPECT(t.device().is_cpu(), "input_layernorm on CPU");
        // The HF model has input_layernorm gamma around ~0.25 (post-training),
        // not the init value of 1.0. We just verify the read is non-zero.
        const auto* p = static_cast<const uint16_t*>(t.data());
        float sum = 0.0f;
        for (int64_t i = 0; i < t.numel(); ++i) {
            // BF16 to float: top 16 bits of FP32
            uint32_t f32_bits = static_cast<uint32_t>(p[i]) << 16;
            float f; std::memcpy(&f, &f32_bits, 4);
            sum += f;
        }
        const float mean = sum / t.numel();
        std::printf("  model.layers.0.input_layernorm.weight mean = %.4f\n", mean);
        // Sanity: nonzero mean and not absurd.
        EXPECT(std::fabs(mean) > 1e-3f && std::fabs(mean) < 10.0f,
               "input_layernorm mean is sane");
    }

    // 4. Missing tensor returns nullptr (no throw).
    {
        const WeightInfo* wi = idx.find("model.does.not.exist");
        EXPECT(wi == nullptr, "missing tensor -> nullptr");
    }

    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}