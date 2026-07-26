#pragma once

#include <cstdint>
#include <vector>

namespace mini_infer {

/**
 * PrefixCache — Week 7/8 placeholder.
 *
 * The real implementation will use a Radix Trie keyed by per-block token
 * hashes, with LRU eviction and copy-on-write on token append. For now,
 * this class is a no-op pass-through used by Scheduler::step() to keep
 * the call sites stable across Week 6 -> Week 7.
 *
 * API (final, but methods are stubs):
 *   - lookup(prompt_ids) -> matched block_count  (0 in Week 6)
 *   - acquire(prompt_ids, BlockTable*)           (no-op in Week 6)
 *   - touch(seq_id, num_blocks)                  (no-op in Week 6)
 *
 * Construction is cheap (no device memory allocated), so every Scheduler
 * can own one without overhead.
 */
class PrefixCache {
public:
    PrefixCache() = default;

    // Returns the number of leading blocks of `prompt_ids` that are
    // already cached and could be reused. Week 6 always returns 0.
    int lookup(const std::vector<int64_t>& /*prompt_ids*/) const { return 0; }

    // Mark a sequence as having used `num_blocks` of cache for prefix
    // sharing. Week 6 is a no-op.
    void acquire(int /*seq_id*/, int /*num_blocks*/) {}

    // Update the LRU timestamp for the given sequence's blocks. Week 6
    // is a no-op.
    void touch(int /*seq_id*/, int /*num_blocks*/) {}

    // Diagnostics — Week 6 always reports zero hits.
    int  total_hits()   const { return 0; }
    int  total_misses() const { return 0; }
};

}  // namespace mini_infer