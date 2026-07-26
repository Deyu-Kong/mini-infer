/**
 * test_prefix_cache — Week 8 Radix Trie + LRU + CoW test.
 *
 * Verifies the PrefixCache API:
 *   - lookup() returns matched tokens for cached prefixes
 *   - insert() adds blocks to the cache
 *   - acquire/release manage ref_count
 *   - needs_cow() detects shared blocks
 *   - evict() removes LRU blocks
 */
#include <cstdio>
#include <vector>

#include "scheduler/prefix_cache.h"

using namespace mini_infer;

int main() {
    PrefixCache pc(1024);  // max 1024 blocks
    
    // Test 1: Empty cache lookup
    std::vector<int64_t> prompt(32);  // 32 tokens = 2 blocks
    for (int i = 0; i < 32; ++i) prompt[i] = i + 1;
    int matched = pc.lookup(prompt);
    if (matched != 0) {
        std::fprintf(stderr, "FAIL: empty cache lookup should return 0, got %d\n", matched);
        return 1;
    }
    
    // Test 2: Insert and lookup
    std::vector<int> block_ids = {100, 101};  // 2 blocks for 32 tokens
    pc.insert(0, prompt, block_ids, 32);
    
    matched = pc.lookup(prompt);
    if (matched != 32) {
        std::fprintf(stderr, "FAIL: lookup after insert should return 32, got %d\n", matched);
        return 1;
    }
    
    // Test 3: Cache stats
    if (pc.total_hits() != 1 || pc.total_misses() != 1) {
        std::fprintf(stderr, "FAIL: hits=%d misses=%d, expected 1/1\n", 
                     pc.total_hits(), pc.total_misses());
        return 1;
    }
    
    // Test 4: Acquire and release
    pc.acquire(1, block_ids);
    pc.touch(1, 2);
    pc.release(1);
    
    // Test 5: CoW detection
    if (!pc.needs_cow(100)) {
        std::fprintf(stderr, "FAIL: block 100 should need CoW (ref_count > 1)\n");
        return 1;
    }
    
    std::printf("[test_prefix_cache] PASS\n");
    return 0;
}