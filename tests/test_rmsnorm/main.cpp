/**
 * test_rmsnorm — verifies our CUDA RMSNorm matches torch reference.
 *
 * Pipeline:
 *   1. Build random FP16 input x [N, D] and weight [D] on CPU.
 *   2. Upload to GPU, run our kernel, download output.
 *   3. Dump inputs + output to /tmp.
 *   4. Invoke Python verifier with same FP16 conversion + torch.rsqrt.
 *   5. Compare latency with cudaEvent.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "layers/rmsnorm.h"
#include "test_helpers.h"

using mini_infer::DType;
using mini_infer::Device;
using mini_infer::Tensor;

static int g_failures = 0;
#define EXPECT(cond, msg)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__,     \
                         msg);                                                \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

// IEEE-754 binary32 -> binary16 conversion (host).
static uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1u;
    uint32_t exp  = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (exp == 0xffu) return static_cast<uint16_t>((sign << 15) | 0x7c00u |
                                                    (mant ? 0x200u : 0));
    int32_t e = static_cast<int32_t>(exp) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>((sign << 15) | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign << 15);
        mant = (mant | 0x800000u) >> (1 - e);
        uint32_t round_bit = 1u << (13 - e);
        uint32_t sticky = round_bit - 1u;
        uint32_t rounded = mant + ((mant >> (14 - e)) & 1u) * round_bit;
        rounded = (rounded + ((rounded & sticky) == round_bit)) >> (13 - e);
        return static_cast<uint16_t>((sign << 15) | (rounded & 0x3ffu));
    }
    uint32_t round_bit = 1u << 12;
    uint32_t sticky = round_bit - 1u;
    uint32_t rounded = mant + ((mant >> 13) & 1u) * round_bit;
    rounded = (rounded + ((rounded & sticky) == round_bit)) >> 13;
    if (rounded & 0x400u) { rounded = 0; e += 1; if (e >= 31) e = 31; }
    return static_cast<uint16_t>((sign << 15) |
                                 (static_cast<uint32_t>(e) << 10) |
                                 (rounded & 0x3ffu));
}

// Convert FP32 CPU tensor to FP16 CPU tensor (without going through GPU).
static Tensor to_f16(const Tensor& t) {
    EXPECT(t.dtype() == DType::FP32, "to_f16 expects FP32");
    Tensor out(t.shape(), DType::FP16, Device::cpu());
    auto* src = static_cast<const float*>(t.data());
    auto* dst = static_cast<uint16_t*>(out.data());
    for (int64_t i = 0; i < t.numel(); ++i) dst[i] = f32_to_f16_bits(src[i]);
    return out;
}

int main() {
    std::printf("mini-infer :: rmsnorm test\n");
    std::printf("--------------------------\n");

    const int N = 4;
    const int D = 4096;
    const float eps = 1e-6f;

    // 1. build inputs on CPU (FP32 for convenience, then cast to FP16)
    Tensor x_f32({N, D}, DType::FP32, Device::cpu());
    mini_infer::fill_uniform(x_f32, 1.0f, /*seed=*/42);
    Tensor w_f32({D}, DType::FP32, Device::cpu());
    mini_infer::fill_uniform(w_f32, 0.5f, /*seed=*/7);

    Tensor x = to_f16(x_f32);
    Tensor w = to_f16(w_f32);

    // 2. run kernel
    mini_infer::RMSNorm norm(D, eps, /*device=*/0);
    norm.set_weight(w);
    Tensor y_dev = norm.forward(x.to(Device::cuda(0)));
    Tensor y     = y_dev.to(Device::cpu());

    // 2b. Gemma-style (1 + weight) variant on the same inputs.
    mini_infer::RMSNorm norm_gemma(D, eps, /*device=*/0, /*add_one=*/true);
    norm_gemma.set_weight(w);
    Tensor y_gemma_dev = norm_gemma.forward(x.to(Device::cuda(0)));
    Tensor y_gemma     = y_gemma_dev.to(Device::cpu());

    // Sanity: the Gemma variant must differ from the standard one (since
    // the multiplier is (1 + w) instead of w), but should still be finite.
    bool differ = false, finite = true;
    {
        const auto* ya = static_cast<const uint16_t*>(y.data());
        const auto* yb = static_cast<const uint16_t*>(y_gemma.data());
        const int64_t total = N * D;
        for (int64_t i = 0; i < total; ++i) {
            if (ya[i] != yb[i]) differ = true;
            // FP16 inf/nan: exponent all 1s.
            const uint16_t bits_a = ya[i];
            const uint16_t bits_b = yb[i];
            if ((bits_a & 0x7c00u) == 0x7c00u) finite = false;
            if ((bits_b & 0x7c00u) == 0x7c00u) finite = false;
        }
    }
    EXPECT(differ, "Gemma (1+w) variant differs from standard (w)");
    EXPECT(finite, "Gemma (1+w) variant produces finite outputs");

    // Dump artifacts (standard variant; existing test compares against torch).
    const std::string dir = "/tmp/mini_infer_rmsnorm";
    std::system(("mkdir -p " + dir).c_str());
    mini_infer::write_bin(dir + "/weight.bin", w);
    mini_infer::write_bin(dir + "/input.bin",  x);
    mini_infer::write_bin(dir + "/our.bin",    y);

    // 4b. Also dump the Gemma variant outputs to a sibling directory and
    // compare against the same torch reference but with the (1 + weight)
    // multiplier.
    const std::string gdir = "/tmp/mini_infer_rmsnorm_gemma";
    std::system(("mkdir -p " + gdir).c_str());
    mini_infer::write_bin(gdir + "/weight.bin", w);
    mini_infer::write_bin(gdir + "/input.bin",  x);
    mini_infer::write_bin(gdir + "/our.bin",    y_gemma);

    // 4. invoke python verifier
    std::string cmd = "python3 tests/verify/verify.py rmsnorm "
        + dir + "/weight.bin " + dir + "/input.bin "
        + " --our-output " + dir + "/our.bin"
        + " --shape " + std::to_string(N) + " " + std::to_string(D)
        + " --eps " + std::to_string(eps)
        + " 2>&1";
    int rc = mini_infer::run_cmd(cmd);
    EXPECT(rc == 0, "matches torch reference");

    // Gemma (1 + weight) variant
    std::string gcmd = "python3 tests/verify/verify.py rmsnorm "
        + gdir + "/weight.bin " + gdir + "/input.bin "
        + " --our-output " + gdir + "/our.bin"
        + " --shape " + std::to_string(N) + " " + std::to_string(D)
        + " --eps " + std::to_string(eps)
        + " --add-one"
        + " 2>&1";
    int grc = mini_infer::run_cmd(gcmd);
    EXPECT(grc == 0, "Gemma (1+w) variant matches torch reference");

    // 5. latency
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    constexpr int WARMUP = 5, ITER = 100;
    for (int i = 0; i < WARMUP; ++i) (void)norm.forward(x.to(Device::cuda(0)));
    cudaEventRecord(s);
    for (int i = 0; i < ITER; ++i) (void)norm.forward(x.to(Device::cuda(0)));
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    std::printf("latency: %.3f us / call  (N=%d, D=%d, ITER=%d)\n",
                ms * 1000.0f / ITER, N, D, ITER);

    cudaEventDestroy(s); cudaEventDestroy(e);
    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}