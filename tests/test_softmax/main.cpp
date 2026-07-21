/**
 * test_softmax — verifies our CUDA softmax matches torch.softmax.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "kernels/softmax_kernel.cuh"
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
    std::printf("mini-infer :: softmax test\n");
    std::printf("--------------------------\n");

    const int N = 8, D = 1024;

    Tensor x_f32({N, D}, DType::FP32, Device::cpu());
    mini_infer::fill_uniform(x_f32, 4.0f, /*seed=*/555);
    Tensor x_f16({N, D}, DType::FP16, Device::cpu());
    {
        auto* src = static_cast<const float*>(x_f32.data());
        auto* dst = static_cast<uint16_t*>(x_f16.data());
        for (int64_t i = 0; i < x_f32.numel(); ++i) dst[i] = f32_to_f16_bits(src[i]);
    }

    Tensor x_dev = x_f16.to(Device::cuda(0));
    Tensor y_dev({N, D}, DType::FP16, Device::cuda(0));

    mini_infer::kernels::launch_softmax(
        static_cast<const __half*>(x_dev.data()),
        static_cast<__half*>(y_dev.data()),
        N, D, /*stream=*/0);
    cudaDeviceSynchronize();

    Tensor y = y_dev.to(Device::cpu());

    const std::string dir = "/tmp/mini_infer_softmax";
    std::system(("mkdir -p " + dir).c_str());
    mini_infer::write_bin(dir + "/input.bin", x_f16);
    mini_infer::write_bin(dir + "/our.bin",   y);

    std::string cmd = "python3 tests/verify/verify.py softmax "
        + dir + "/input.bin "
        + " --our-output " + dir + "/our.bin"
        + " --shape " + std::to_string(N) + " " + std::to_string(D);
    int rc = mini_infer::run_cmd(cmd);
    EXPECT(rc == 0, "matches torch reference");

    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}