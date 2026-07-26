/**
 * test_prefix_cache — Week 6 placeholder test.
 *
 * Verifies the no-op API compiles and behaves correctly:
 *   - lookup() always returns 0 (no caching yet).
 *   - acquire/touch are silent no-ops.
 *   - total_hits/total_misses always 0.
 *
 * In Week 7/8 this test will be extended to validate the real Radix Trie.
 */
#include <cstdio>
#include <vector>

#include "scheduler/prefix_cache.h"

using namespace mini_infer;

int main() {
    PrefixCache pc;
    if (pc.lookup({1, 2, 3, 4, 5}) != 0) {
        std::fprintf(stderr, "FAIL: lookup should return 0 in Week 6\n");
        return 1;
    }
    if (pc.total_hits() != 0 || pc.total_misses() != 0) {
        std::fprintf(stderr, "FAIL: hits/misses should be 0\n");
        return 1;
    }
    pc.acquire(0, 4);  // no-op, must not crash
    pc.touch(0, 4);
    std::printf("[test_prefix_cache] PASS\n");
    return 0;
}