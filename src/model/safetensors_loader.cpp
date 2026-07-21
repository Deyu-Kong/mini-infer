#include "model/safetensors_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace mini_infer {

namespace {

DType parse_dtype(const std::string& s) {
    // Common safetensors dtype strings. Anything else falls back to FP32.
    if (s == "F32" || s == "float32") return DType::FP32;
    if (s == "F16" || s == "float16" || s == "half")  return DType::FP16;
    if (s == "BF16" || s == "bfloat16") return DType::BF16;
    if (s == "I32") return DType::INT32;
    if (s == "I64") return DType::INT64;
    return DType::FP32;
}

std::size_t file_size_of(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("SafetensorsLoader: stat failed on " + path);
    }
    return static_cast<std::size_t>(st.st_size);
}

}  // namespace

// ----------------------------------------------------------------------------
// SafetensorsLoader
// ----------------------------------------------------------------------------
SafetensorsLoader::SafetensorsLoader(const std::string& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("SafetensorsLoader: open failed on " + path);
    }
    file_size_ = file_size_of(path);
    if (file_size_ < 8) {
        throw std::runtime_error("SafetensorsLoader: file too small: " + path);
    }
    void* p = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("SafetensorsLoader: mmap failed on " + path);
    }
    mmap_ptr_ = p;

    // First 8 bytes = little-endian uint64 header length.
    uint64_t header_length_le = 0;
    std::memcpy(&header_length_le, mmap_ptr_, sizeof(uint64_t));
    header_length_ = static_cast<std::size_t>(header_length_le);

    if (8 + header_length_ > file_size_) {
        throw std::runtime_error("SafetensorsLoader: header length exceeds file size: " + path);
    }
    data_section_ = static_cast<const uint8_t*>(mmap_ptr_) + 8 + header_length_;
    data_size_    = file_size_ - 8 - header_length_;

    parse_header_();
}

SafetensorsLoader::~SafetensorsLoader() {
    if (mmap_ptr_) ::munmap(mmap_ptr_, file_size_);
    if (fd_ >= 0)  ::close(fd_);
}

void SafetensorsLoader::parse_header_() {
    const char* json_begin = static_cast<const char*>(mmap_ptr_) + 8;
    std::string json_str(json_begin, header_length_);

    nlohmann::json j = nlohmann::json::parse(json_str);
    if (j.is_object() == false) {
        throw std::runtime_error("SafetensorsLoader: header JSON is not an object: " + path_);
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& name = it.key();
        if (name == "__metadata__") continue;  // HF metadata, not a tensor

        const auto& v = it.value();
        if (!v.contains("dtype") || !v.contains("shape") ||
            !v.contains("data_offsets")) {
            throw std::runtime_error("SafetensorsLoader: tensor missing fields: " + name);
        }
        WeightInfo wi;
        wi.dtype = parse_dtype(v["dtype"].get<std::string>());
        for (auto d : v["shape"]) wi.shape.push_back(d.get<int64_t>());
        const auto offs = v["data_offsets"];
        if (!offs.is_array() || offs.size() != 2) {
            throw std::runtime_error("SafetensorsLoader: bad data_offsets for " + name);
        }
        const int64_t begin = offs[0].get<int64_t>();
        const int64_t end   = offs[1].get<int64_t>();
        wi.byte_offset = static_cast<std::size_t>(begin);
        wi.byte_size   = static_cast<std::size_t>(end - begin);
        const std::size_t expect = static_cast<std::size_t>(wi.numel()) *
                                    dtype_size(wi.dtype);
        if (wi.byte_size != expect) {
            throw std::runtime_error("SafetensorsLoader: byte size mismatch for " +
                                     name);
        }
        tensors_.emplace(name, wi);
    }
}

const WeightInfo* SafetensorsLoader::find(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

Tensor SafetensorsLoader::read_to_cpu(const std::string& name) const {
    const WeightInfo* wi = find(name);
    if (!wi) {
        throw std::runtime_error("SafetensorsLoader::read_to_cpu: missing " + name);
    }
    Tensor t(wi->shape, wi->dtype, Device::cpu());
    std::memcpy(t.data(), data_section_ + wi->byte_offset, wi->byte_size);
    return t;
}

// ----------------------------------------------------------------------------
// WeightIndex
// ----------------------------------------------------------------------------
WeightIndex WeightIndex::load(const std::string& shards_root) {
    WeightIndex idx;
    idx.shards_root_ = shards_root;

    // Read index.json (optional — single-shard models won't have one).
    const std::string index_path = shards_root + "/model.safetensors.index.json";
    std::ifstream f(index_path);
    bool has_index = f.good();
    if (has_index) {
        nlohmann::json j; f >> j;
        const auto& wm = j.at("weight_map");
        for (auto it = wm.begin(); it != wm.end(); ++it) {
            idx.weight_map_.emplace(it.key(), it.value().get<std::string>());
        }
        idx.total_tensors_ = idx.weight_map_.size();
    } else {
        // Single-shard model — try the obvious filename.
        const std::string single = shards_root + "/model.safetensors";
        if (file_size_of(single) > 0) {
            idx.shards_.emplace("model.safetensors",
                                std::make_pair(single, std::make_unique<SafetensorsLoader>(single)));
        } else {
            throw std::runtime_error(
                "WeightIndex: no model.safetensors.index.json and no "
                "model.safetensors found in " + shards_root);
        }
    }

    // Lazy-open shards on first access via loader_for().
    return idx;
}

const SafetensorsLoader* WeightIndex::loader_for(const std::string& name) const {
    std::string shard_name;
    auto it = weight_map_.find(name);
    if (it != weight_map_.end()) {
        shard_name = it->second;
    } else if (!shards_.empty()) {
        shard_name = shards_.begin()->first;  // single-shard fallback
    } else {
        return nullptr;
    }
    auto sit = shards_.find(shard_name);
    if (sit == shards_.end()) {
        const std::string full = shards_root_ + "/" + shard_name;
        auto loader = std::make_unique<SafetensorsLoader>(full);
        sit = shards_.emplace(shard_name,
                              std::make_pair(full, std::move(loader))).first;
    }
    return sit->second.second.get();
}

const WeightInfo* WeightIndex::find(const std::string& name) const {
    const SafetensorsLoader* l = loader_for(name);
    if (!l) return nullptr;
    return l->find(name);
}

Tensor WeightIndex::read_to_cpu(const std::string& name) const {
    const SafetensorsLoader* l = loader_for(name);
    if (!l) throw std::runtime_error("WeightIndex: missing tensor " + name);
    return l->read_to_cpu(name);
}

}  // namespace mini_infer