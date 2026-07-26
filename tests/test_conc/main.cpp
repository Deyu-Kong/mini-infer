
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "core/allocator.h"
#include "scheduler/kv_cache.h"
#include "scheduler/paged_kv_cache.h"

int main() {
    using mini_infer::BlockAllocator;
    using mini_infer::KVCache;
    using mini_infer::PagedKVCache;
    const int NUM_LAYERS = 1;
    const int NUM_KV_HEADS = 2;
    const int HEAD_DIM = 64;
    const int MAX_SEQ = 256;

    // Naive: pre-allocates max_seq per sequence * layers.
    KVCache naive(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ, 0);
    // We can only fit ONE sequence in the naive cache (single contiguous
    // region per layer). So naive_concurrent = 1.
    std::printf("naive_concurrent=1 (single contiguous block per layer)\n");

    // Paged: pool of N blocks. Each sequence needs ceil(L / BS) blocks.
    const int num_pool = 1024;
    PagedKVCache paged(num_pool, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM,
                       /*max_blocks_per_seq=*/MAX_SEQ, 0);
    // Try to host as many sequences of length L as possible.
    const int L = 32;     // 2 blocks per sequence
    int paged_concurrent = 0;
    int sid = 0;
    while (true) {
        paged.create_sequence(sid);
        bool ok = true;
        for (int t = 0; t < L; ++t) {
            if (paged.append_token(sid) < 0) {
                ok = false; break;
            }
        }
        if (!ok) {
            paged.destroy_sequence(sid);
            break;
        }
        ++paged_concurrent;
        ++sid;
    }
    std::printf("paged_concurrent=%d (length=%d, pool=%d blocks)\n",
                paged_concurrent, L, num_pool);

    double speedup = static_cast<double>(paged_concurrent) / 1.0;
    std::printf("speedup=%.2fx\n", speedup);
    return 0;
}
