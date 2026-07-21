/**
 * test_mlp — full SwiGLU MLP forward (cuBLAS GEMMs + fused SwiGLU).
 *
 * Compares against:
 *   y = (silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
 * via torch.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "layers/mlp.h"
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

static void cast_to_f16(const Tensor& f32, Tensor& f16) {
    auto* src = static_cast<const float*>(f32.data());
    auto* dst = static_cast<uint16_t*>(f16.data());
    for (int64_t i = 0; i < f32.numel(); ++i) dst[i] = f32_to_f16_bits(src[i]);
}

int main() {
    std::printf("mini-infer :: mlp test\n");
    std::printf("----------------------\n");

    const int B = 4, H = 512, I = 1376;  // arbitrary; matches a Qwen2.5 shape

    Tensor x_f32  ({B, H}, DType::FP32, Device::cpu());
    Tensor wg_f32 ({I, H}, DType::FP32, Device::cpu());
    Tensor wu_f32 ({I, H}, DType::FP32, Device::cpu());
    Tensor wd_f32 ({H, I}, DType::FP32, Device::cpu());
    mini_infer::fill_uniform(x_f32,   0.5f, 1);
    mini_infer::fill_uniform(wg_f32,  0.05f, 2);
    mini_infer::fill_uniform(wu_f32,  0.05f, 3);
    mini_infer::fill_uniform(wd_f32,  0.05f, 4);

    Tensor x_f16  ({B, H}, DType::FP16, Device::cpu());
    Tensor wg_f16 ({I, H}, DType::FP16, Device::cpu());
    Tensor wu_f16 ({I, H}, DType::FP16, Device::cpu());
    Tensor wd_f16 ({H, I}, DType::FP16, Device::cpu());
    cast_to_f16(x_f32,  x_f16);
    cast_to_f16(wg_f32, wg_f16);
    cast_to_f16(wu_f32, wu_f16);
    cast_to_f16(wd_f32, wd_f16);

    mini_infer::MLP mlp(H, I, /*device=*/0);
    mlp.set_weights(wg_f16, wu_f16, wd_f16);

    Tensor y_dev = mlp.forward(x_f16.to(Device::cuda(0)));
    Tensor y     = y_dev.to(Device::cpu());

    const std::string dir = "/tmp/mini_infer_mlp";
    std::system(("mkdir -p " + dir).c_str());
    mini_infer::write_bin(dir + "/x.bin",  x_f16);
    mini_infer::write_bin(dir + "/wg.bin", wg_f16);
    mini_infer::write_bin(dir + "/wu.bin", wu_f16);
    mini_infer::write_bin(dir + "/wd.bin", wd_f16);
    mini_infer::write_bin(dir + "/our.bin", y);

    std::string cmd = "python3 tests/verify/verify.py mlp "
        + dir + "/x.bin " + dir + "/wg.bin " + dir + "/wu.bin " + dir + "/wd.bin "
        + " --our-output " + dir + "/our.bin --shape "
        + std::to_string(B) + " " + std::to_string(H) + " " + std::to_string(I);
    int rc = mini_infer::run_cmd(cmd);
    EXPECT(rc == 0, "matches torch reference");

    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}