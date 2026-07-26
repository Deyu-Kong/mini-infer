#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/allocator.h"
#include "core/tensor.h"

namespace mini_infer {

/**
 * Per-request block table — the indirection layer between logical
 * sequence position and physical KV block.
 *
 * For a sequence with `num_tokens` tokens, `block_ids.size() ==
 * ceil(num_tokens / BLOCK_SIZE)` blocks are required. Position `s`
 * lives in block `block_ids[s / BLOCK_SIZE]` at offset `s % BLOCK_SIZE`
 * within that block.
 */
struct BlockTable {
    std::vector<int> block_ids;
    int              num_tokens = 0;   // logical sequence length
};

/**
 * PagedKVCache — Week 5 PagedAttention KV storage.
 *
 * Owns a BlockAllocator (one pool shared across all layers) and a
 * per-request BlockTable map. New sequences register via
 * `create_sequence(seq_id)`; KV pairs for token `s` of sequence `seq_id`
 * are written into block `block_table.block_ids[s / BLOCK_SIZE]`,
 * offset `s % BLOCK_SIZE`.
 *
 * Sequence lifecycle:
 *   create_sequence(sid)        -> empty BlockTable
 *   append_token(sid)           -> grow block table by 1 token (alloc if needed)
 *   k_ptr_for(sid, layer, s)    -> __half* to K[layer, block, kv_head, offset, :]
 *   v_ptr_for(sid, layer, s)    -> same for V
 *   destroy_sequence(sid)       -> free all blocks, remove table
 *
 * Concurrency:
 *   The block allocator is single-threaded (one host thread per
 *   Engine). Sequences are independent — they may have different
 *   lengths and grow at different rates.
 */
class PagedKVCache {
public:
    // Convenience alias for callers that need the block size without
    // pulling in the allocator header. Equals BlockAllocator::kBlockSize.
    static constexpr int kBlockSizeHint = 16;

    PagedKVCache(int num_blocks,
                 int num_layers,
                 int num_kv_heads,
                 int head_dim,
                 int max_blocks_per_seq,
                 int device_index);

    int  create_sequence(int seq_id);
    void destroy_sequence(int seq_id);

    // Destroy every sequence currently registered in the cache and free
    // all blocks back to the pool. Used by benchmarks to reset between
    // independent runs. Week 6 helper.
    void clear_all_sequences();

    // Append a new token (grows the sequence by 1). Returns the new
    // logical position (== num_tokens before append) or -1 on OOM.
    int append_token(int seq_id);

    // Rollback the sequence to a previous token count. Frees any blocks
    // that become completely empty. Used by speculative decoding when
    // draft tokens are rejected.
    //   new_num_tokens: target token count (must be <= current num_tokens)
    void rollback(int seq_id, int new_num_tokens);

    // K / V storage pointer for sequence `seq_id`, layer `layer`, kv
    // position `token_pos` (in [0, num_tokens)).  Caller is responsible
    // for not writing past `head_dim` elements per token.
    void* k_ptr_for(int seq_id, int layer, int token_pos);
    void* v_ptr_for(int seq_id, int layer, int token_pos);

    // Direct access to the block table (needed by the paged-attention
    // kernel — it walks block_ids on device).
    const BlockTable& table(int seq_id) const;
    int               num_tokens(int seq_id) const;

    // Diagnostic: how many blocks does this sequence currently hold?
    int num_blocks(int seq_id) const;

    // Pool-wide stats (test diagnostics).
    int total_in_use_blocks() const { return allocator_.in_use_blocks(); }
    int total_free_blocks()   const { return allocator_.num_free_blocks(); }
    int total_num_blocks()   const { return allocator_.num_blocks(); }
    int num_active_sequences() const { return static_cast<int>(tables_.size()); }

    // Underlying allocator (useful for the attention layer to query dims).
    const BlockAllocator& allocator() const { return allocator_; }
    int max_blocks_per_seq() const { return max_blocks_per_seq_; }

private:
    BlockAllocator allocator_;
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    int max_blocks_per_seq_;
    int device_index_;

    std::unordered_map<int, BlockTable> tables_;
};

}  // namespace mini_infer