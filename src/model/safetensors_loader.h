#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"

namespace mini_infer {

/**
 * One tensor as recorded in a safetensors file's JSON header.
 *
 *  - `dtype`           : how the bytes are encoded (FP32 / FP16 / BF16).
 *  - `shape`           : tensor shape (row-major semantics).
 *  - `byte_offset`     : absolute byte offset from the start of the
 *                        *data section* (i.e. past the JSON header).
 *  - `byte_size`       : byte size = numel * dtype_size.
 */
struct WeightInfo {
    DType       dtype = DType::FP32;
    std::vector<int64_t> shape;
    std::size_t byte_offset = 0;
    std::size_t byte_size   = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t s : shape) n *= s;
        return n;
    }
};

/**
 * SafetensorsLoader — memory-maps a single .safetensors shard.
 *
 * File layout (per the format spec):
 *
 *   [0 .. 7]                  : little-endian uint64 N = header JSON length
 *   [8 .. 8+N)                : UTF-8 JSON header
 *   [8+N .. end)              : tensor data (interleaved per data_offsets)
 *
 * The JSON maps each tensor name to {dtype, shape, data_offsets}. Offsets
 * are absolute byte positions within the data section.
 *
 * We mmap the whole file so the GPU `cudaMemcpy` can DMA straight out of
 * the page cache without inflating RSS.
 *
 * Multi-shard models should be opened with one loader per file and their
 * tensors merged into a `WeightIndex`.
 */
class SafetensorsLoader {
public:
    explicit SafetensorsLoader(const std::string& path);
    ~SafetensorsLoader();

    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;

    const std::string& path()        const { return path_; }
    const uint8_t*     data_section() const { return data_section_; }
    std::size_t        data_size()    const { return data_size_; }

    const WeightInfo* find(const std::string& name) const;

    // Iterators over the tensors (header order is preserved by nlohmann::json).
    const std::unordered_map<std::string, WeightInfo>& tensors() const {
        return tensors_;
    }

    // Convenience: read a tensor's bytes (still in original dtype) into a
    // freshly-allocated CPU tensor. Used by the model loader to copy a few
    // small parameters (e.g. RMSNorm gamma) onto the host.
    Tensor read_to_cpu(const std::string& name) const;

private:
    void parse_header_();

    std::string  path_;
    int          fd_ = -1;
    void*        mmap_ptr_ = nullptr;
    std::size_t  file_size_ = 0;
    std::size_t  header_length_ = 0;
    const uint8_t* data_section_ = nullptr;
    std::size_t  data_size_ = 0;
    std::unordered_map<std::string, WeightInfo> tensors_;
};

/**
 * WeightIndex — aggregates tensors from multiple safetensors shards
 * (the standard HF format for models that don't fit in a single file).
 *
 * Resolution order: each name is looked up in its assigned shard via the
 * weight_map loaded from `model.safetensors.index.json`.
 */
class WeightIndex {
public:
    // `shards_root` is the directory containing model-00001-of-N.safetensors;
    // it must also contain model.safetensors.index.json.
    static WeightIndex load(const std::string& shards_root);

    const WeightInfo* find(const std::string& name) const;
    const SafetensorsLoader* loader_for(const std::string& name) const;

    // Read a tensor to CPU across all shards.
    Tensor read_to_cpu(const std::string& name) const;

    std::size_t num_tensors() const { return total_tensors_; }
    const std::string& shards_root() const { return shards_root_; }

private:
    std::string shards_root_;
    mutable std::unordered_map<std::string,
                       std::pair<std::string, std::unique_ptr<SafetensorsLoader>>>
        shards_;   // shard filename -> loader (mutable for lazy loading)
    std::unordered_map<std::string, std::string> weight_map_;  // tensor name -> shard
    std::size_t total_tensors_ = 0;
};

}  // namespace mini_infer