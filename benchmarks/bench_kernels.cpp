/**
 * bench_kernels — measure single-kernel latency with cudaEvent.
 *
 * Each kernel is run N=200 times; the median and p99 are reported.
 * Sizes chosen to roughly match Qwen2.5 layer dimensions.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/tensor.h"
#include "kernels/rmsnorm_kernel.cuh"
#include "kernels/rope_kernel.cuh"
#include "kernels/softmax_kernel.cuh"
#include "kernels/swiglu_kernel.cuh"

using mini_infer::DType;
using mini_infer::Device;
using mini_infer::Tensor;

#define MI_CUDA_CHECK(call)                                                   \
    do {                                                                      \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %d\n", #call, __LINE__);    \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

template <typename Fn>
void bench(const char* name, Fn fn, int warmup = 10, int iters = 200) {
    std::vector<float> times(iters);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(s);
        fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&times[i], s, e);
    }
    std::sort(times.begin(), times.end());
    const float med = times[iters / 2];
    const float p99 = times[(iters * 99) / 100];
    std::printf("  %-30s  median=%7.2f us   p99=%7.2f us\n",
                name, med * 1000.f, p99 * 1000.f);
    cudaEventDestroy(s); cudaEventDestroy(e);
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    // RMSNorm : typical Qwen2.5 layer dims (B=4 tokens, D=3584 hidden)
    {
        const int N = 4, D = 3584;
        Tensor x({N, D}, DType::FP16, Device::cuda(dev));
        Tensor w({D}, DType::FP16, Device::cuda(dev));
        Tensor y({N, D}, DType::FP16, Device::cuda(dev));
        auto fn = [&](){
            mini_infer::kernels::launch_rmsnorm(
                static_cast<const __half*>(x.data()),
                static_cast<const __half*>(w.data()),
                static_cast<__half*>(y.data()), N, D, 1e-6f,
                /*add_one=*/0, /*stream=*/0);
        };
        char name[64]; std::snprintf(name, sizeof(name),
                                     "rmsnorm N=%d D=%d", N, D);
        bench(name, fn);
    }

    // RoPE : typical prefill B=4, S=512, H=28, D=128 (Qwen2.5 7B head_dim)
    {
        const int B = 4, S = 512, H = 28, D = 128;
        const int half = D / 2;
        Tensor x({B, S, H, D}, DType::FP16, Device::cuda(dev));
        Tensor ct({S, half}, DType::FP16, Device::cuda(dev));
        Tensor st({S, half}, DType::FP16, Device::cuda(dev));
        Tensor y({B, S, H, D}, DType::FP16, Device::cuda(dev));
        std::vector<int64_t> pos(S); for (int i = 0; i < S; ++i) pos[i] = i;
        std::vector<float> inv(half, 1.0f);
        auto fn = [&](){
            mini_infer::kernels::launch_rope_precompute(
                inv.data(), pos.data(),
                static_cast<__half*>(ct.data()),
                static_cast<__half*>(st.data()), S, half, 0);
            mini_infer::kernels::launch_rope(
                static_cast<const __half*>(x.data()),
                static_cast<const __half*>(ct.data()),
                static_cast<const __half*>(st.data()),
                static_cast<__half*>(y.data()), B, S, H, D, 0);
        };
        char name[64]; std::snprintf(name, sizeof(name),
                                     "rope B=%d S=%d H=%d D=%d", B, S, H, D);
        bench(name, fn);
    }

    // Softmax : typical attention row count (B=4, S=512, H=28 -> 4*512 rows of 512)
    {
        const int N = 4 * 28 * 512, D = 512;
        Tensor x({N, D}, DType::FP16, Device::cuda(dev));
        Tensor y({N, D}, DType::FP16, Device::cuda(dev));
        auto fn = [&](){
            mini_infer::kernels::launch_softmax(
                static_cast<const __half*>(x.data()),
                static_cast<__half*>(y.data()), N, D, 0);
        };
        char name[64]; std::snprintf(name, sizeof(name),
                                     "softmax N=%d D=%d", N, D);
        bench(name, fn);
    }

    // SwiGLU : typical MLP intermediate dim I=18944 (Qwen2.5 7B intermediate)
    {
        const int N = 4 * 512, I = 18944;
        Tensor g({N, I}, DType::FP16, Device::cuda(dev));
        Tensor u({N, I}, DType::FP16, Device::cuda(dev));
        Tensor o({N, I}, DType::FP16, Device::cuda(dev));
        auto fn = [&](){
            mini_infer::kernels::launch_swiglu(
                static_cast<const __half*>(g.data()),
                static_cast<const __half*>(u.data()),
                static_cast<__half*>(o.data()), N * I, 0);
        };
        char name[64]; std::snprintf(name, sizeof(name),
                                     "swiglu N=%d I=%d", N, I);
        bench(name, fn);
    }

    return 0;
}