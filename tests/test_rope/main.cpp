/**
 * test_rope — verifies RoPE matches torch reference on FP16 input.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "layers/rope.h"
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

static uint16_t f32_to_f16_bits(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
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

int main() {
    std::printf("mini-infer :: rope test\n");
    std::printf("-----------------------\n");

    const int B = 2, S = 16, H = 4, D = 64;
    mini_infer::RoPE rope(D, /*theta=*/1000000.0f, /*device=*/0);

    // build input
    Tensor x_f32({B, S, H, D}, DType::FP32, Device::cpu());
    mini_infer::fill_uniform(x_f32, 1.0f, /*seed=*/101);

    Tensor x_fp16({B, S, H, D}, DType::FP16, Device::cpu());
    {
        auto* src = static_cast<const float*>(x_f32.data());
        auto* dst = static_cast<uint16_t*>(x_fp16.data());
        for (int64_t i = 0; i < x_f32.numel(); ++i)
            dst[i] = f32_to_f16_bits(src[i]);
    }

    // positions
    std::vector<int64_t> positions(S);
    mini_infer::fill_uniform_int64(positions, /*seed=*/2024, 0, 4096);

    // run kernel
    Tensor y_dev = rope.forward(x_fp16.to(Device::cuda(0)), positions);
    Tensor y     = y_dev.to(Device::cpu());

    // We need to dump x (FP16), cos, sin tables (FP16), positions (int64),
    // and our output (FP16). Python verifier uses the same cos/sin tables.
    // Easiest: recompute cos/sin on host and dump them along with positions.
    const int half = D / 2;
    Tensor cos_t_f16({S, half}, DType::FP16, Device::cpu());
    Tensor sin_t_f16({S, half}, DType::FP16, Device::cpu());
    for (int s = 0; s < S; ++s) {
        for (int i = 0; i < half; ++i) {
            const float angle = static_cast<float>(positions[s]) * rope.inv_freq()[i];
            const uint16_t cb = f32_to_f16_bits(std::cos(angle));
            const uint16_t sb = f32_to_f16_bits(std::sin(angle));
            static_cast<uint16_t*>(cos_t_f16.data())[s * half + i] = cb;
            static_cast<uint16_t*>(sin_t_f16.data())[s * half + i] = sb;
        }
    }

    // dump
    const std::string dir = "/tmp/mini_infer_rope";
    std::system(("mkdir -p " + dir).c_str());
    mini_infer::write_bin(dir + "/x.bin",    x_fp16);
    mini_infer::write_bin(dir + "/cos.bin",  cos_t_f16);
    mini_infer::write_bin(dir + "/sin.bin",  sin_t_f16);
    {
        std::ofstream f(dir + "/pos.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(positions.data()),
                positions.size() * sizeof(int64_t));
    }
    mini_infer::write_bin(dir + "/our.bin", y);

    std::string cmd = "python3 tests/verify/verify.py rope "
        + dir + "/x.bin " + dir + "/cos.bin " + dir + "/sin.bin "
        + dir + "/pos.bin "
        + " --our-output " + dir + "/our.bin"
        + " --shape " + std::to_string(B) + " " + std::to_string(S) + " "
        + std::to_string(H) + " " + std::to_string(D);
    int rc = mini_infer::run_cmd(cmd);
    EXPECT(rc == 0, "matches torch reference");

    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}