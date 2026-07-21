/**
 * test_tensor — verifies Tensor round-trip (host<->device) plus a
 * trivial add kernel that exercises device-side computation.
 *
 * Checks:
 *   1. shape / numel / nbytes / dtype reporting
 *   2. contiguous stride computation
 *   3. h2d / d2h copy preserves data exactly
 *   4. device-side add produces the expected sum
 */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/tensor.h"

using mini_infer::DType;
using mini_infer::Device;
using mini_infer::Tensor;

#define MI_CUDA_CHECK(call)                                                   \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d : %s\n",             \
                         #call, __FILE__, __LINE__,                           \
                         cudaGetErrorString(_e));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

static int g_failures = 0;
#define EXPECT(cond, msg)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__,     \
                         msg);                                                \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

static void test_metadata() {
    std::printf("[1] metadata + strides ... ");
    Tensor t({2, 3, 4}, DType::FP32, Device::cpu());
    EXPECT(t.ndim() == 3, "ndim");
    EXPECT(t.numel() == 24, "numel");
    EXPECT(t.nbytes() == 24 * 4, "nbytes");
    EXPECT(t.shape().size() == 3, "shape size");
    EXPECT(t.is_contiguous(), "row-major contiguous");
    // Row-major contiguous strides for [2,3,4]: [12, 4, 1]
    EXPECT(t.stride()[0] == 12 && t.stride()[1] == 4 && t.stride()[2] == 1,
           "stride values");
    EXPECT(t.dtype() == DType::FP32, "dtype");
    EXPECT(t.device().is_cpu(), "device");
    std::printf("ok\n");
}

static void test_fill_cpu() {
    std::printf("[2] cpu fill FP32 ... ");
    Tensor t({4}, DType::FP32, Device::cpu());
    t.fill(3.5f);
    auto* p = static_cast<float*>(t.data());
    for (int i = 0; i < 4; ++i) EXPECT(p[i] == 3.5f, "fill value");
    std::printf("ok\n");
}

static void test_fill_cuda_zero() {
    std::printf("[3] cuda memset to zero ... ");
    Tensor t({8}, DType::FP32, Device::cuda(0));
    t.fill(0.0f);
    std::vector<float> h(8);
    MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data(), 8 * sizeof(float),
                             cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) EXPECT(h[i] == 0.0f, "zero");
    std::printf("ok\n");
}

static void test_h2d_d2h_roundtrip() {
    std::printf("[4] h2d / d2h round-trip (FP32, N=1024) ... ");
    constexpr int N = 1024;
    Tensor cpu({N}, DType::FP32, Device::cpu());
    auto* hp = static_cast<float*>(cpu.data());
    for (int i = 0; i < N; ++i) hp[i] = static_cast<float>(i) * 0.25f;

    Tensor gpu = cpu.to(Device::cuda(0));
    EXPECT(gpu.device().is_cuda(), "to(cuda) device");
    EXPECT(gpu.numel() == N, "numel preserved");
    EXPECT(gpu.dtype() == DType::FP32, "dtype preserved");

    Tensor back = gpu.to(Device::cpu());
    auto* bp = static_cast<float*>(back.data());
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        if (hp[i] != bp[i]) ++bad;
    }
    EXPECT(bad == 0, "round-trip values");
    std::printf("ok\n");
}

static void test_add_kernel() {
    std::printf("[5] device-side add kernel ... ");
    constexpr int N = 4096;
    Tensor a({N}, DType::FP32, Device::cpu());
    Tensor b({N}, DType::FP32, Device::cpu());
    auto* ap = static_cast<float*>(a.data());
    auto* bp = static_cast<float*>(b.data());
    for (int i = 0; i < N; ++i) { ap[i] = i * 1.0f; bp[i] = i * 2.0f; }

    Tensor a_dev = a.to(Device::cuda(0));
    Tensor b_dev = b.to(Device::cuda(0));
    Tensor c_dev({N}, DType::FP32, Device::cuda(0));

    constexpr int threads = 256;
    constexpr int blocks  = (N + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(
        static_cast<const float*>(a_dev.data()),
        static_cast<const float*>(b_dev.data()),
        static_cast<float*>(c_dev.data()),
        N);
    MI_CUDA_CHECK(cudaGetLastError());
    MI_CUDA_CHECK(cudaDeviceSynchronize());

    Tensor c = c_dev.to(Device::cpu());
    auto* cp = static_cast<float*>(c.data());
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        const float expect = ap[i] + bp[i];
        if (std::fabs(cp[i] - expect) > 1e-5f) ++bad;
    }
    EXPECT(bad == 0, "a + b == expect");
    std::printf("ok\n");
}

int main() {
    std::printf("mini-infer :: tensor tests\n");
    std::printf("---------------------------\n");
    test_metadata();
    test_fill_cpu();
    test_fill_cuda_zero();
    test_h2d_d2h_roundtrip();
    test_add_kernel();
    if (g_failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d test(s) failed.\n", g_failures);
    return 1;
}