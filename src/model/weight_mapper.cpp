#include "model/weight_mapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "core/dtype_utils.h"

namespace mini_infer {

namespace {

// Convert a CPU tensor to FP16 (same device). Supports BF16 / FP32 inputs.
Tensor convert_cpu_to_f16(const Tensor& src) {
    if (src.dtype() == DType::FP16) return src;
    Tensor dst(src.shape(), DType::FP16, Device::cpu());
    const int64_t n = src.numel();
    if (src.dtype() == DType::BF16) {
        const auto* s = static_cast<const uint16_t*>(src.data());
        auto*       d = static_cast<uint16_t*>(dst.data());
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t f32_bits = static_cast<uint32_t>(s[i]) << 16;
            float f; std::memcpy(&f, &f32_bits, 4);
            d[i] = f32_to_f16_bits(f);
        }
        return dst;
    }
    if (src.dtype() == DType::FP32) {
        const auto* s = static_cast<const float*>(src.data());
        auto*       d = static_cast<uint16_t*>(dst.data());
        for (int64_t i = 0; i < n; ++i) d[i] = f32_to_f16_bits(s[i]);
        return dst;
    }
    throw std::runtime_error("convert_cpu_to_f16: unsupported dtype " +
                             std::string(dtype_name(src.dtype())));
}

// Throw if `idx.find(name)` returns nullptr.
const WeightInfo& require_weight(const WeightIndex& idx,
                                 const std::string& name) {
    const WeightInfo* wi = idx.find(name);
    if (!wi) {
        throw std::runtime_error("WeightNameMapper: missing weight " + name);
    }
    return *wi;
}

void check_shape(const Tensor& t,
                 const std::vector<int64_t>& want,
                 const std::string& name) {
    if (t.shape() != want) {
        std::string got, exp;
        for (size_t i = 0; i < t.shape().size(); ++i) {
            if (i) got += ",";
            got += std::to_string(t.shape()[i]);
        }
        for (size_t i = 0; i < want.size(); ++i) {
            if (i) exp += ",";
            exp += std::to_string(want[i]);
        }
        throw std::runtime_error("WeightNameMapper: bad shape for " + name +
                                 " got [" + got + "] expected [" + exp + "]");
    }
}

}  // namespace

Tensor WeightNameMapper::load_weight_as_f16(const WeightIndex& idx,
                                            const std::string& name,
                                            int device_index) {
    require_weight(idx, name);
    Tensor cpu = idx.read_to_cpu(name);
    Tensor cpu_f16 = convert_cpu_to_f16(cpu);
    return cpu_f16.to(Device::cuda(device_index));
}

WeightNameMapper::QKVLoadResult
WeightNameMapper::load_qkv(const WeightIndex& idx,
                           const LayerWeightNames& names,
                           int64_t num_heads,
                           int64_t num_kv_heads,
                           int64_t head_dim,
                           int64_t hidden_size,
                           int device_index) const {
    QKVLoadResult out;

    if (arch_ == ModelArch::Gemma && !names.qkv_proj.empty()
        && idx.find(names.qkv_proj) != nullptr) {
        // Gemma with merged qkv_proj (rare; some Gemma 1 checkpoints).
        // One merged tensor of shape [num_heads*head_dim +
        //                              2*num_kv_heads*head_dim,
        //                              hidden_size].
        const int64_t q_rows = num_heads    * head_dim;
        const int64_t kv_rows = num_kv_heads * head_dim;
        const int64_t total_rows = q_rows + 2 * kv_rows;
        Tensor qkv = load_weight_as_f16(idx, names.qkv_proj, device_index);
        check_shape(qkv, {total_rows, hidden_size}, names.qkv_proj);

        auto* src = static_cast<const __half*>(qkv.data());

        out.w_q = Tensor::empty({q_rows,  hidden_size}, DType::FP16,
                                 Device::cuda(device_index));
        out.w_k = Tensor::empty({kv_rows, hidden_size}, DType::FP16,
                                 Device::cuda(device_index));
        out.w_v = Tensor::empty({kv_rows, hidden_size}, DType::FP16,
                                 Device::cuda(device_index));

        const size_t row_bytes = static_cast<size_t>(hidden_size) * sizeof(__half);
        cudaMemcpy2DAsync(out.w_q.data(), row_bytes,
                          src + 0,                              row_bytes,
                          row_bytes, q_rows,
                          cudaMemcpyDeviceToDevice, /*stream=*/0);
        cudaMemcpy2DAsync(out.w_k.data(), row_bytes,
                          src + q_rows * hidden_size,           row_bytes,
                          row_bytes, kv_rows,
                          cudaMemcpyDeviceToDevice, /*stream=*/0);
        cudaMemcpy2DAsync(out.w_v.data(), row_bytes,
                          src + (q_rows + kv_rows) * hidden_size, row_bytes,
                          row_bytes, kv_rows,
                          cudaMemcpyDeviceToDevice, /*stream=*/0);
        cudaDeviceSynchronize();
        // Gemma has no bias.
        return out;
    }

    // Default path: three separate projections (Qwen / LLaMA / Mistral /
    // Yi / DeepSeek / Gemma 2 / Gemma 3 with separate QKV).
    out.w_q = load_weight_as_f16(idx, names.q_proj, device_index);
    out.w_k = load_weight_as_f16(idx, names.k_proj, device_index);
    out.w_v = load_weight_as_f16(idx, names.v_proj, device_index);
    check_shape(out.w_q, {num_heads    * head_dim, hidden_size}, names.q_proj);
    check_shape(out.w_k, {num_kv_heads * head_dim, hidden_size}, names.k_proj);
    check_shape(out.w_v, {num_kv_heads * head_dim, hidden_size}, names.v_proj);
    // Bias is optional; the loader checks the field before calling.
    return out;
}

}  // namespace mini_infer