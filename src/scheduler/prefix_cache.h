#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_infer {

/**
 * PrefixCache — Radix Trie based prefix caching with LRU eviction.
 *
 * Week 8: Full implementation with:
 *   - Radix Trie indexed by per-block token hashes
 *   - LRU eviction when cache exceeds capacity
 *   - Copy-on-write semantics for shared blocks
 *
 * The cache stores KV blocks keyed by token sequence prefixes. When a new
 * request arrives, we lookup the longest matching prefix and reuse those
 * blocks, avoiding redundant computation.
 *
 * Block hash: Each block of BLOCK_SIZE tokens is hashed (SHA256 truncated
 * to 64-bit) to form the trie key. This allows efficient prefix matching
 * without storing full token sequences.
 *
 * LRU: When cache is full, evict least-recently-used blocks. Blocks with
 * ref_count > 1 are skipped (still in use by active sequences).
 *
 * CoW: When a sequence appends to a shared block (ref_count > 1), allocate
 * a new block and copy the KV data. This prevents mutations from affecting
 * other sequences sharing the same prefix.
 */
class PrefixCache {
public:
    static constexpr int kBlockSize = 16;
    static constexpr int kDefaultMaxBlocks = 1024;
    static constexpr float kEvictionThreshold = 0.3f;  // evict when > 70% full

    explicit PrefixCache(int max_blocks = kDefaultMaxBlocks);
    ~PrefixCache() = default;

    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    /**
     * Lookup the longest matching prefix for prompt_ids.
     * Returns the number of tokens that can be reused from cache.
     * Also populates matched_blocks with the block IDs to reuse.
     */
    int lookup(const std::vector<int64_t>& prompt_ids,
               std::vector<int>* matched_blocks = nullptr);

    /**
     * Insert a sequence's blocks into the cache after generation completes.
     * The blocks are now available for future requests to reuse.
     *   seq_id: sequence identifier
     *   prompt_ids: the prompt tokens (used to compute block hashes)
     *   block_ids: physical block IDs to cache
     *   num_tokens: total tokens in the sequence
     */
    void insert(int seq_id,
                const std::vector<int64_t>& prompt_ids,
                const std::vector<int>& block_ids,
                int num_tokens);

    /**
     * Mark blocks as in-use by a sequence. Increments ref_count.
     */
    void acquire(int seq_id, const std::vector<int>& block_ids);

    /**
     * Release blocks when a sequence completes. Decrements ref_count.
     * Blocks with ref_count == 0 become candidates for eviction.
     */
    void release(int seq_id);

    /**
     * Update LRU timestamp for a sequence's blocks (on each token append).
     */
    void touch(int seq_id, int num_blocks);

    /**
     * Check if a block needs copy-on-write.
     * Returns true if block_id has ref_count > 1 (shared by multiple sequences).
     */
    bool needs_cow(int block_id) const;

    /**
     * Evict LRU blocks to free up space. Called when cache is near capacity.
     * Returns number of blocks evicted.
     */
    int evict(int target_free_blocks);

    // Diagnostics
    int total_hits() const { return total_hits_; }
    int total_misses() const { return total_misses_; }
    int cached_blocks() const { return cached_blocks_; }
    int max_blocks() const { return max_blocks_; }
    double hit_rate() const {
        int total = total_hits_ + total_misses_;
        return total > 0 ? static_cast<double>(total_hits_) / total : 0.0;
    }

private:
    struct TrieNode {
        std::unordered_map<uint64_t, std::shared_ptr<TrieNode>> children;
        int block_id = -1;  // physical block ID if this is a leaf
        int ref_count = 0;  // number of sequences using this block
        uint64_t last_access = 0;  // LRU timestamp
    };

    struct SequenceInfo {
        std::vector<int> block_ids;  // blocks owned by this sequence
        uint64_t last_access = 0;
    };

    // Hash a block of tokens (BLOCK_SIZE tokens)
    static uint64_t hash_block(const int64_t* tokens, int count);

    // Split prompt into blocks and compute hashes
    std::vector<uint64_t> compute_block_hashes(const std::vector<int64_t>& tokens) const;

    std::shared_ptr<TrieNode> root_;
    std::unordered_map<int, SequenceInfo> sequences_;
    std::list<std::pair<int, uint64_t>> lru_list_;  // (block_id, timestamp)
    
    int max_blocks_;
    int cached_blocks_ = 0;
    int total_hits_ = 0;
    int total_misses_ = 0;
    uint64_t timestamp_ = 0;
};

}  // namespace mini_infer
