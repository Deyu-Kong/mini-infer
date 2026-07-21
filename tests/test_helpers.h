#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor.h"

namespace mini_infer {

// Write a CPU FP16 / FP32 tensor's raw bytes to a binary file. The
// shape/dtype are captured separately by the caller (typically via a JSON
// config or --shape CLI flag for the python verify script).
inline void write_bin(const std::string& path, const Tensor& t) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_bin: cannot open " + path);
    f.write(reinterpret_cast<const char*>(t.data()),
            static_cast<std::streamsize>(t.nbytes()));
    if (!f) throw std::runtime_error("write_bin: write failed " + path);
}

// Read raw bytes into a freshly-allocated CPU FP16 tensor of the given shape.
inline Tensor read_bin_f16(const std::string& path,
                            const std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("read_bin: cannot open " + path);
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    const std::size_t expect = static_cast<std::size_t>(numel) * 2;  // FP16
    if (static_cast<std::size_t>(bytes) != expect) {
        throw std::runtime_error("read_bin: size mismatch " + path);
    }
    Tensor t(shape, DType::FP16, Device::cpu());
    f.read(reinterpret_cast<char*>(t.data()), bytes);
    if (!f) throw std::runtime_error("read_bin: read failed " + path);
    return t;
}

// Fill a CPU FP32 tensor with uniform random values in [-scale, scale].
inline void fill_uniform(Tensor& t, float scale, uint32_t seed) {
    if (t.dtype() != DType::FP32) {
        throw std::runtime_error("fill_uniform expects FP32 tensor");
    }
    uint32_t x = seed ? seed : 0x9E3779B9u;
    auto* p = static_cast<float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        const float u = (static_cast<float>((x >> 1) & 0x7FFFFFFF) /
                         static_cast<float>(0x80000000)) * 2.0f - 1.0f;
        p[i] = u * scale;
    }
}

inline void fill_uniform_int64(std::vector<int64_t>& v, uint32_t seed,
                                int64_t lo, int64_t hi) {
    uint32_t x = seed ? seed : 0x12345678u;
    for (size_t i = 0; i < v.size(); ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        const int64_t range = hi - lo + 1;
        v[i] = lo + static_cast<int64_t>((x >> 1) % static_cast<uint32_t>(range));
    }
}

// Run `cmd` (a shell command) and throw if exit code != 0.
inline int run_cmd(const std::string& cmd) {
    int rc = std::system(cmd.c_str());
    return rc;
}

}  // namespace mini_infer