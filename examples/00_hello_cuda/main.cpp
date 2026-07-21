/**
 * 00_hello_cuda — minimal sanity-check program.
 *
 * Prints device info (name, SM count, memory, compute capability),
 * launches a trivial kernel that fills a buffer with its thread id,
 * and verifies the kernel actually ran on the GPU.
 *
 * Run:  ./build/examples/00_hello_cuda
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

__global__ void hello_kernel(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx * idx;
}

int main() {
    int ndev = 0;
    MI_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    std::printf("=========================================\n");
    std::printf(" mini-infer : hello CUDA  (Week 1)\n");
    std::printf("=========================================\n");
    std::printf("Detected %d CUDA device(s)\n\n", ndev);
    if (ndev == 0) {
        std::fprintf(stderr, "No CUDA device available.\n");
        return 1;
    }

    std::vector<cudaDeviceProp> props(ndev);
    for (int i = 0; i < ndev; ++i) {
        MI_CUDA_CHECK(cudaGetDeviceProperties(&props[i], i));
    }

    int target = 0;
    MI_CUDA_CHECK(cudaSetDevice(target));
    const auto& p = props[target];

    char driver[64] = {0};
    int runtime_ver = 0;
    MI_CUDA_CHECK(cudaDriverGetVersion(reinterpret_cast<int*>(driver)));
    // Above stores the version in the int's bytes; decode properly:
    {
        int v = 0;
        MI_CUDA_CHECK(cudaDriverGetVersion(&v));
        runtime_ver = v;
    }

    int rt = 0;
    MI_CUDA_CHECK(cudaRuntimeGetVersion(&rt));

    std::printf("Using device %d : %s\n", target, p.name);
    std::printf("  Compute capability : %d.%d\n", p.major, p.minor);
    std::printf("  SM count           : %d\n", p.multiProcessorCount);
    std::printf("  Total global mem   : %.2f GiB\n",
                static_cast<double>(p.totalGlobalMem) / (1ull << 30));
    std::printf("  Driver version     : %d.%d\n",
                runtime_ver / 1000, (runtime_ver % 1000) / 10);
    std::printf("  Runtime version    : %d.%d\n",
                rt / 1000, (rt % 1000) / 10);
    std::printf("  Warp size          : %d\n", p.warpSize);
    std::printf("\n");

    // Launch the trivial kernel and verify.
    constexpr int N = 1024;
    int* d_buf = nullptr;
    MI_CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(int)));
    constexpr int threads = 128;
    constexpr int blocks  = (N + threads - 1) / threads;
    hello_kernel<<<blocks, threads>>>(d_buf, N);
    MI_CUDA_CHECK(cudaGetLastError());
    MI_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h(N);
    MI_CUDA_CHECK(cudaMemcpy(h.data(), d_buf, N * sizeof(int),
                             cudaMemcpyDeviceToHost));
    MI_CUDA_CHECK(cudaFree(d_buf));

    int bad = 0;
    for (int i = 0; i < N; ++i) {
        if (h[i] != i * i) ++bad;
    }
    std::printf("Kernel roundtrip: %d / %d correct\n", N - bad, N);
    if (bad != 0) {
        std::fprintf(stderr, "Kernel produced wrong results.\n");
        return 2;
    }
    std::printf("OK\n");
    return 0;
}