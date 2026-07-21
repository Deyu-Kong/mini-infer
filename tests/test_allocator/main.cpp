/**
 * test_allocator — BumpAllocator behavior checks.
 *
 *  1. allocations are 256-byte aligned
 *  2. monotonic cursor: back-to-back allocs never overlap
 *  3. reset() returns the cursor to zero (memory stays mapped)
 *  4. over-capacity allocation returns nullptr
 *  5. touch the returned pointer with a kernel launch to confirm
 *     the underlying device memory is actually usable
 */
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/allocator.h"

using mini_infer::BumpAllocator;

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

__global__ void touch_kernel(int* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = i + 1;
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

int main() {
    std::printf("mini-infer :: allocator tests\n");
    std::printf("-----------------------------\n");

    constexpr std::size_t cap = 16ull * 1024 * 1024;  // 16 MiB
    BumpAllocator alloc(cap, 0);

    EXPECT(alloc.capacity() == cap, "capacity");
    EXPECT(alloc.used() == 0, "fresh used=0");

    // [1] alignment
    std::printf("[1] 256-byte alignment ... ");
    void* p1 = alloc.allocate(1);
    void* p2 = alloc.allocate(257);
    void* p3 = alloc.allocate(4096);
    EXPECT(p1 != nullptr && p2 != nullptr && p3 != nullptr, "allocate ok");
    EXPECT((reinterpret_cast<uintptr_t>(p1) & 0xFFu) == 0, "p1 aligned");
    EXPECT((reinterpret_cast<uintptr_t>(p2) & 0xFFu) == 0, "p2 aligned");
    EXPECT((reinterpret_cast<uintptr_t>(p3) & 0xFFu) == 0, "p3 aligned");
    std::printf("ok\n");

    // [2] monotonic, no overlap
    std::printf("[2] monotonic, no overlap ... ");
    const std::size_t used_before = alloc.used();
    void* a = alloc.allocate(1024);
    void* b = alloc.allocate(1024);
    EXPECT(static_cast<char*>(b) - static_cast<char*>(a) >= 1024, "b after a");
    EXPECT(alloc.used() > used_before, "used increased");
    std::printf("ok\n");

    // [3] touch with a kernel (writes device memory via the bump pool)
    std::printf("[3] GPU write via pool ... ");
    constexpr int N = 1024;
    int* dev = static_cast<int*>(alloc.allocate(N * sizeof(int)));
    EXPECT(dev != nullptr, "allocate N ints");
    touch_kernel<<<4, 256>>>(dev, N);
    MI_CUDA_CHECK(cudaGetLastError());
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> host(N);
    MI_CUDA_CHECK(cudaMemcpy(host.data(), dev, N * sizeof(int),
                             cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; ++i) if (host[i] != i + 1) ++bad;
    EXPECT(bad == 0, "kernel wrote through pool");
    std::printf("ok\n");

    // [4] reset()
    std::printf("[4] reset() ... ");
    const std::size_t peak = alloc.peak();
    EXPECT(peak > 0, "peak tracked");
    alloc.reset();
    EXPECT(alloc.used() == 0, "used reset to 0");
    EXPECT(alloc.peak() == peak, "peak preserved");
    void* p4 = alloc.allocate(64);
    EXPECT(p4 != nullptr, "post-reset allocate");
    std::printf("ok\n");

    // [5] over-capacity returns nullptr
    std::printf("[5] over-capacity ... ");
    BumpAllocator tiny(1024, 0);
    void* ok = tiny.allocate(1000);
    EXPECT(ok != nullptr, "1000 fits in 1024");
    void* overflow = tiny.allocate(2000);
    EXPECT(overflow == nullptr, "2000 overflows");
    std::printf("ok\n");

    if (g_failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d test(s) failed.\n", g_failures);
    return 1;
}