#include "scheduler/prefix_cache.h"

#include <algorithm>
#include <cstring>
#include <functional>

namespace mini_infer {

PrefixCache::PrefixCache(int max_blocks)
    : root_(std::make_shared<TrieNode>()),
      max_blocks_(max_blocks) {}

uint64_t PrefixCache::hash_block(const int64_t* tokens, int count) {
    // FNV-1a hash (64-bit) - fast and good distribution
    uint64_t hash = 14695981039346656037ULL;
    for (int i = 0; i < count; ++i) {
        hash ^= static_cast<uint64_t>(tokens[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::vector<uint64_t> PrefixCache::compute_block_hashes(
    const std::vector<int64_t>& tokens) const {
    std::vector<uint64_t> hashes;
    int num_blocks = static_cast<int>(tokens.size()) / kBlockSize;
    
    for (int i = 0; i < num_blocks; ++i) {
        uint64_t h = hash_block(tokens.data() + i * kBlockSize, kBlockSize);
        hashes.push_back(h);
    }
    
    return hashes;
}

int PrefixCache::lookup(const std::vector<int64_t>& prompt_ids,
                        std::vector<int>* matched_blocks) {
    std::vector<uint64_t> block_hashes = compute_block_hashes(prompt_ids);
    
    if (block_hashes.empty()) {
        ++total_misses_;
        return 0;
    }
    
    auto node = root_;
    int matched_tokens = 0;
    
    for (uint64_t h : block_hashes) {
        auto it = node->children.find(h);
        if (it == node->children.end()) {
            break;
        }
        node = it->second;
        if (node->block_id >= 0) {
            if (matched_blocks) {
                matched_blocks->push_back(node->block_id);
            }
            matched_tokens += kBlockSize;
            node->last_access = ++timestamp_;
        }
    }
    
    if (matched_tokens > 0) {
        ++total_hits_;
    } else {
        ++total_misses_;
    }
    
    return matched_tokens;
}

void PrefixCache::insert(int seq_id,
                         const std::vector<int64_t>& prompt_ids,
                         const std::vector<int>& block_ids,
                         int num_tokens) {
    std::vector<uint64_t> block_hashes = compute_block_hashes(prompt_ids);
    
    if (block_hashes.empty() || block_ids.empty()) {
        return;
    }
    
    // Check if we need to evict
    if (cached_blocks_ + static_cast<int>(block_ids.size()) > max_blocks_) {
        int target_free = static_cast<int>(block_ids.size()) + 
                          static_cast<int>(max_blocks_ * kEvictionThreshold);
        evict(target_free);
    }
    
    auto node = root_;
    int num_to_cache = std::min(static_cast<int>(block_hashes.size()),
                                static_cast<int>(block_ids.size()));
    
    for (int i = 0; i < num_to_cache; ++i) {
        uint64_t h = block_hashes[i];
        
        auto it = node->children.find(h);
        if (it == node->children.end()) {
            // Insert new node
            auto new_node = std::make_shared<TrieNode>();
            new_node->block_id = block_ids[i];
            new_node->ref_count = 1;
            new_node->last_access = ++timestamp_;
            node->children[h] = new_node;
            node = new_node;
            ++cached_blocks_;
            
            // Add to LRU list
            lru_list_.push_back({block_ids[i], timestamp_});
        } else {
            // Node exists, update ref_count
            node = it->second;
            ++node->ref_count;
            node->last_access = ++timestamp_;
        }
    }
    
    // Track sequence
    SequenceInfo info;
    info.block_ids = block_ids;
    info.last_access = timestamp_;
    sequences_[seq_id] = info;
}

void PrefixCache::acquire(int seq_id, const std::vector<int>& block_ids) {
    // Increment ref_count for all blocks by searching the trie
    std::function<void(const std::shared_ptr<TrieNode>&)> increment_refs = 
        [&](const std::shared_ptr<TrieNode>& node) {
        for (const auto& [h, child] : node->children) {
            for (int block_id : block_ids) {
                if (child->block_id == block_id) {
                    ++child->ref_count;
                    child->last_access = ++timestamp_;
                    break;
                }
            }
            increment_refs(child);
        }
    };
    increment_refs(root_);
    
    SequenceInfo info;
    info.block_ids = block_ids;
    info.last_access = ++timestamp_;
    sequences_[seq_id] = info;
}

void PrefixCache::release(int seq_id) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) {
        return;
    }
    
    // Decrement ref_count for all blocks
    // In a full implementation, we'd traverse the trie and decrement
    // For now, just remove from sequence tracking
    sequences_.erase(it);
}

void PrefixCache::touch(int seq_id, int num_blocks) {
    auto it = sequences_.find(seq_id);
    if (it != sequences_.end()) {
        it->second.last_access = ++timestamp_;
    }
}

bool PrefixCache::needs_cow(int block_id) const {
    // Search trie for this block_id and check ref_count
    // Linear search - could optimize with reverse map
    std::function<bool(const std::shared_ptr<TrieNode>&)> search = 
        [&](const std::shared_ptr<TrieNode>& node) -> bool {
        if (node->block_id == block_id) {
            return node->ref_count > 1;
        }
        for (const auto& [h, child] : node->children) {
            if (search(child)) return true;
        }
        return false;
    };
    
    return search(root_);
}

int PrefixCache::evict(int target_free_blocks) {
    int evicted = 0;
    
    // Sort LRU list by timestamp (oldest first)
    lru_list_.sort([](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    
    auto it = lru_list_.begin();
    while (it != lru_list_.end() && evicted < target_free_blocks) {
        int block_id = it->first;
        
        // Find and remove from trie
        std::function<bool(const std::shared_ptr<TrieNode>&)> remove_block = 
            [&](const std::shared_ptr<TrieNode>& node) -> bool {
            for (auto child_it = node->children.begin(); 
                 child_it != node->children.end(); ++child_it) {
                if (child_it->second->block_id == block_id && 
                    child_it->second->ref_count <= 1) {
                    node->children.erase(child_it);
                    --cached_blocks_;
                    return true;
                }
                if (remove_block(child_it->second)) {
                    return true;
                }
            }
            return false;
        };
        
        if (remove_block(root_)) {
            it = lru_list_.erase(it);
            ++evicted;
        } else {
            ++it;  // Skip blocks still in use
        }
    }
    
    return evicted;
}

}  // namespace mini_infer
