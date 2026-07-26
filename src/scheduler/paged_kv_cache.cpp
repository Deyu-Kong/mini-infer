#include "scheduler/paged_kv_cache.h"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace mini_infer {

namespace {
inline void check_seq_id(int seq_id, const std::unordered_map<int, BlockTable>& t) {
    if (t.find(seq_id) == t.end()) {
        throw std::runtime_error("PagedKVCache: unknown seq_id " +
                                 std::to_string(seq_id));
    }
}
}  // namespace

PagedKVCache::PagedKVCache(int num_blocks, int num_layers, int num_kv_heads,
                           int head_dim, int max_blocks_per_seq,
                           int device_index)
    : allocator_(num_blocks, num_layers, num_kv_heads, head_dim, device_index),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      max_blocks_per_seq_(max_blocks_per_seq),
      device_index_(device_index) {
    if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0
        || max_blocks_per_seq <= 0) {
        throw std::runtime_error("PagedKVCache: invalid dimensions");
    }
}

int PagedKVCache::create_sequence(int seq_id) {
    if (tables_.count(seq_id)) {
        throw std::runtime_error("PagedKVCache::create_sequence: duplicate id");
    }
    BlockTable t;
    t.num_tokens = 0;
    tables_.emplace(seq_id, std::move(t));
    return seq_id;
}

void PagedKVCache::destroy_sequence(int seq_id) {
    auto it = tables_.find(seq_id);
    if (it == tables_.end()) return;
    for (int b : it->second.block_ids) {
        allocator_.free(b);
    }
    tables_.erase(it);
}

void PagedKVCache::clear_all_sequences() {
    for (auto& kv : tables_) {
        for (int b : kv.second.block_ids) {
            allocator_.free(b);
        }
    }
    tables_.clear();
}

int PagedKVCache::append_token(int seq_id) {
    auto it = tables_.find(seq_id);
    if (it == tables_.end()) {
        throw std::runtime_error("PagedKVCache::append_token: unknown seq_id");
    }
    BlockTable& t = it->second;
    const int pos = t.num_tokens;
    const int needed_blocks = (pos / BlockAllocator::kBlockSize) + 1;
    if (needed_blocks > max_blocks_per_seq_) {
        throw std::runtime_error("PagedKVCache: max_blocks_per_seq exceeded");
    }
    if (pos % BlockAllocator::kBlockSize == 0) {
        const int b = allocator_.alloc();
        if (b < 0) return -1;          // OOM
        t.block_ids.push_back(b);
    }
    ++t.num_tokens;
    return pos;
}

void PagedKVCache::rollback(int seq_id, int new_num_tokens) {
    auto it = tables_.find(seq_id);
    if (it == tables_.end()) {
        throw std::runtime_error("PagedKVCache::rollback: unknown seq_id");
    }
    BlockTable& t = it->second;
    if (new_num_tokens > t.num_tokens) {
        throw std::runtime_error("PagedKVCache::rollback: cannot grow");
    }
    if (new_num_tokens < 0) {
        new_num_tokens = 0;
    }
    
    // Calculate how many blocks we need for new_num_tokens
    const int needed_blocks = (new_num_tokens + BlockAllocator::kBlockSize - 1) 
                              / BlockAllocator::kBlockSize;
    
    // Free blocks that are no longer needed
    while (static_cast<int>(t.block_ids.size()) > needed_blocks) {
        int b = t.block_ids.back();
        t.block_ids.pop_back();
        allocator_.free(b);
    }
    
    t.num_tokens = new_num_tokens;
}

void* PagedKVCache::k_ptr_for(int seq_id, int layer, int token_pos) {
    auto it = tables_.find(seq_id);
    if (it == tables_.end()) {
        throw std::runtime_error("PagedKVCache::k_ptr_for: unknown seq_id");
    }
    const BlockTable& t = it->second;
    if (token_pos < 0 || token_pos >= t.num_tokens) {
        throw std::runtime_error("PagedKVCache::k_ptr_for: token_pos out of range");
    }
    const int block_idx  = token_pos / BlockAllocator::kBlockSize;
    const int token_off  = token_pos % BlockAllocator::kBlockSize;
    void* blk = allocator_.k_block_ptr(layer, t.block_ids[block_idx]);
    // Layout: [num_kv_heads, BLOCK_SIZE, head_dim] -> stride inside block:
    //   token_off * num_kv_heads * head_dim + kv_head * head_dim + d
    // Caller (Attention::forward_paged) does the kv_head * head_dim add.
    return static_cast<__half*>(blk) + token_off * num_kv_heads_ * head_dim_;
}

void* PagedKVCache::v_ptr_for(int seq_id, int layer, int token_pos) {
    auto it = tables_.find(seq_id);
    if (it == tables_.end()) {
        throw std::runtime_error("PagedKVCache::v_ptr_for: unknown seq_id");
    }
    const BlockTable& t = it->second;
    if (token_pos < 0 || token_pos >= t.num_tokens) {
        throw std::runtime_error("PagedKVCache::v_ptr_for: token_pos out of range");
    }
    const int block_idx = token_pos / BlockAllocator::kBlockSize;
    const int token_off = token_pos % BlockAllocator::kBlockSize;
    void* blk = allocator_.v_block_ptr(layer, t.block_ids[block_idx]);
    return static_cast<__half*>(blk) + token_off * num_kv_heads_ * head_dim_;
}

const BlockTable& PagedKVCache::table(int seq_id) const {
    return tables_.at(seq_id);
}

int PagedKVCache::num_tokens(int seq_id) const {
    auto it = tables_.find(seq_id);
    if (it == tables_.end()) return 0;
    return it->second.num_tokens;
}

int PagedKVCache::num_blocks(int seq_id) const {
    auto it = tables_.find(seq_id);
    if (it == tables_.end()) return 0;
    return static_cast<int>(it->second.block_ids.size());
}

}  // namespace mini_infer