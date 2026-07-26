/**
 * test_allocator — BumpAllocator + BlockAllocator behavior checks.
 *
 *  BumpAllocator:
 *  1. allocations are 256-byte aligned
 *  2. monotonic cursor: back-to-back allocs never overlap
 *  3. reset() returns the cursor to zero (memory stays mapped)
 *  4. over-capacity allocation returns nullptr
 *  5. touch the returned pointer with a kernel launch to confirm
 *     the underlying device memory is actually usable
 *
 *  BlockAllocator (Week 5):
 *  6. initial state (all free, none in use)
 *  7. alloc/free round trip + ref count semantics
 *  8. OOM when pool exhausted
 *  9. K / V block storage is actually writable
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/allocator.h"

using mini_infer::BumpAllocator;
using mini_infer::BlockAllocator;

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

__global__ void touch_h_kernel(__half* p, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2half(v);
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

    // ====================================================================
    // BlockAllocator (Week 5, PagedAttention)
    // ====================================================================
    using mini_infer::BlockAllocator;
    std::printf("\n[Week 5] BlockAllocator tests\n");
    std::printf("-----------------------------\n");

    constexpr int NUM_BLOCKS   = 64;
    constexpr int NUM_LAYERS   = 2;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM     = 16;          // small for fast tests
    BlockAllocator ba(NUM_BLOCKS, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, 0);

    // [6] free list starts full
    std::printf("[6] initial state ... ");
    EXPECT(ba.num_free_blocks() == NUM_BLOCKS, "all blocks free");
    EXPECT(ba.in_use_blocks() == 0, "none in use");
    EXPECT(ba.num_blocks() == NUM_BLOCKS, "num_blocks");
    std::printf("ok\n");

    // [7] alloc/free round trip, ref count tracking
    std::printf("[7] alloc/free round trip ... ");
    int b0 = ba.alloc();
    int b1 = ba.alloc();
    EXPECT(b0 >= 0 && b1 >= 0 && b0 != b1, "two distinct blocks");
    EXPECT(ba.num_free_blocks() == NUM_BLOCKS - 2, "two fewer free");
    EXPECT(ba.ref_count(b0) == 1, "b0 ref=1");
    EXPECT(ba.ref_count(b1) == 1, "b1 ref=1");
    // Sharing (prefix-cache style): ref twice.
    ba.ref(b0);
    EXPECT(ba.ref_count(b0) == 2, "b0 ref=2 after ref");
    ba.unref(b0);
    EXPECT(ba.ref_count(b0) == 1, "b0 back to 1");
    ba.free(b0);
    EXPECT(ba.num_free_blocks() == NUM_BLOCKS - 1, "one back on free list");
    ba.free(b1);
    EXPECT(ba.num_free_blocks() == NUM_BLOCKS, "all free again");
    std::printf("ok\n");

    // [8] OOM when pool exhausted
    std::printf("[8] OOM on full pool ... ");
    std::vector<int> ids;
    while (true) {
        int x = ba.alloc();
        if (x < 0) break;
        ids.push_back(x);
    }
    EXPECT(static_cast<int>(ids.size()) == NUM_BLOCKS, "filled the pool");
    EXPECT(ba.num_free_blocks() == 0, "no free left");
    int x = ba.alloc();
    EXPECT(x < 0, "alloc returns -1 when full");
    for (int id : ids) ba.free(id);
    EXPECT(ba.num_free_blocks() == NUM_BLOCKS, "refilled after free");
    std::printf("ok\n");

    // [9] block storage is writable
    std::printf("[9] K / V block storage writable ... ");
    int bb = ba.alloc();
    auto* kptr = static_cast<__half*>(ba.k_block_ptr(0, bb));
    auto* vptr = static_cast<__half*>(ba.v_block_ptr(1, bb));
    EXPECT(kptr != nullptr, "k_ptr not null");
    EXPECT(vptr != nullptr, "v_ptr not null");
    EXPECT(kptr != vptr, "K and V are separate buffers");
    constexpr int N_PER_BLOCK = NUM_KV_HEADS * BlockAllocator::kBlockSize * HEAD_DIM;
    const int threads_needed = 256;
    const int blocks_needed = (N_PER_BLOCK + threads_needed - 1) / threads_needed;
    touch_h_kernel<<<blocks_needed, threads_needed>>>(kptr, N_PER_BLOCK, 0.5f);
    MI_CUDA_CHECK(cudaGetLastError());
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__half> h_k(N_PER_BLOCK);
    MI_CUDA_CHECK(cudaMemcpy(h_k.data(), kptr, N_PER_BLOCK * sizeof(__half),
                             cudaMemcpyDeviceToHost));
    int bad_h = 0;
    for (int i = 0; i < N_PER_BLOCK; ++i) if (__half2float(h_k[i]) != 0.5f) ++bad_h;
    EXPECT(bad_h == 0, "kernel wrote K block correctly");
    ba.free(bb);
    std::printf("ok\n");

    if (g_failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d test(s) failed.\n", g_failures);
    return 1;
}