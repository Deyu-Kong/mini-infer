
#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <vector>
#include "core/allocator.h"
#include "scheduler/paged_kv_cache.h"

int main() {
    using mini_infer::BlockAllocator;
    using mini_infer::PagedKVCache;
    // 100 sequences, lengths uniformly in [1, 200] tokens.
    const int N_SEQ = 100;
    const int MAX_LEN = 200;
    const int NUM_KV_HEADS = 2;
    const int HEAD_DIM = 64;
    const int NUM_LAYERS = 1;

    // Pick pool size so total tokens ~= sum / 2 (overcommit by 50% to
    // simulate realistic pressure).
    std::mt19937 rng(42);
    std::vector<int> lengths(N_SEQ);
    long long sum = 0;
    for (int i = 0; i < N_SEQ; ++i) {
        lengths[i] = 1 + rng() % MAX_LEN;
        sum += lengths[i];
    }
    const int total_blocks_needed = (sum + BlockAllocator::kBlockSize - 1)
                                    / BlockAllocator::kBlockSize;
    const int num_pool_blocks = total_blocks_needed / 2 + 8;

    PagedKVCache cache(num_pool_blocks, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM,
                       /*max_blocks_per_seq=*/MAX_LEN, /*device=*/0);

    std::vector<int> sids;
    long long total_logical_tokens = 0;
    long long total_blocks_in_use = 0;
    int rejected = 0;
    for (int i = 0; i < N_SEQ; ++i) {
        const int sid = 1000 + i;
        cache.create_sequence(sid);
        sids.push_back(sid);
        int appended = 0;
        for (int t = 0; t < lengths[i]; ++t) {
            if (cache.append_token(sid) < 0) {
                ++rejected;
                break;
            }
            ++appended;
        }
        total_logical_tokens += appended;
        total_blocks_in_use += cache.num_blocks(sid);
    }
    long long total_capacity = static_cast<long long>(total_blocks_in_use)
                               * BlockAllocator::kBlockSize;
    double waste_ratio = 1.0 - static_cast<double>(total_logical_tokens)
                                  / static_cast<double>(total_capacity);
    std::printf("sequences=%d  total_tokens=%lld  blocks_in_use=%lld\n",
                N_SEQ, total_logical_tokens, total_blocks_in_use);
    std::printf("waste_ratio=%.4f\n", waste_ratio);
    std::printf("rejected=%d (pool OOM)\n", rejected);

    // Tear down: free all sequences.
    for (int sid : sids) cache.destroy_sequence(sid);
    std::printf("after teardown: free=%d in_use=%d\n",
                cache.total_free_blocks(),
                cache.total_in_use_blocks());
    return 0;
}
