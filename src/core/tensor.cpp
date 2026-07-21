#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace mini_infer {

const char* dtype_name(DType d) {
    switch (d) {
        case DType::FP32:  return "fp32";
        case DType::FP16:  return "fp16";
        case DType::BF16:  return "bf16";
        case DType::INT32: return "int32";
        case DType::INT64: return "int64";
    }
    return "unknown";
}

namespace {
void cuda_check_(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                                 std::to_string(line) + " : " +
                                 cudaGetErrorString(err));
    }
}
#define MI_CUDA_CHECK(call) cuda_check_(call, __FILE__, __LINE__)
}  // namespace

// Exposed helper so weight loaders can convert BF16 -> FP16 on CPU.
uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1u;
    uint32_t exp  = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (exp == 0xffu) {
        // Inf / NaN
        return static_cast<uint16_t>((sign << 15) | 0x7c00u |
                                     (mant ? 0x200u : 0));
    }
    int32_t e = static_cast<int32_t>(exp) - 127 + 15;
    if (e >= 31) {
        // Overflow -> Inf
        return static_cast<uint16_t>((sign << 15) | 0x7c00u);
    }
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign << 15);
        mant = (mant | 0x800000u) >> (1 - e);
        // Round-to-nearest-even
        uint32_t round_bit = 1u << (13 - e);
        uint32_t sticky = round_bit - 1u;
        uint32_t rounded = mant + ((mant >> (14 - e)) & 1u) * round_bit;
        rounded = (rounded + ((rounded & sticky) == round_bit)) >> (13 - e);
        return static_cast<uint16_t>((sign << 15) | (rounded & 0x3ffu));
    }
    uint32_t round_bit = 1u << 12;
    uint32_t sticky = round_bit - 1u;
    uint32_t rounded = mant + ((mant >> 13) & 1u) * round_bit;
    rounded = (rounded + ((rounded & sticky) == round_bit)) >> 13;
    if (rounded & 0x400u) { rounded = 0; e += 1; if (e >= 31) e = 31; }
    return static_cast<uint16_t>((sign << 15) |
                                 (static_cast<uint32_t>(e) << 10) |
                                 (rounded & 0x3ffu));
}

Tensor::Tensor() = default;

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device) {
    compute_default_stride_();
    numel_ = 1;
    for (int64_t s : shape_) numel_ *= s;
    if (numel_ < 0) numel_ = 0;  // empty tensor
    nbytes_ = static_cast<std::size_t>(numel_) * dtype_size(dtype_);
    allocate_();
}

Tensor Tensor::empty(std::vector<int64_t> shape, DType dtype, Device device) {
    return Tensor(std::move(shape), dtype, device);
}

Tensor::Tensor(Tensor&& other) noexcept { *this = std::move(other); }

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      stride_(other.stride_),
      numel_(other.numel_),
      nbytes_(other.nbytes_),
      dtype_(other.dtype_),
      device_(other.device_),
      data_(nullptr),
      owns_data_(false) {
    // Allocate fresh memory on the same device and copy bytes. The mmap'd
    // safetensors regions are read-only so we *must* copy if we want to
    // take ownership of a view; the deep copy also makes the engine robust
    // against the source being munmap'd underneath us.
    allocate_();
    if (nbytes_ > 0 && data_ != nullptr && other.data_ != nullptr) {
        copy_raw(other.data_, data_, nbytes_, other.device_, device_);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        Tensor tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release_();
        shape_ = std::move(other.shape_);
        stride_ = std::move(other.stride_);
        numel_ = other.numel_;
        nbytes_ = other.nbytes_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.owns_data_ = false;
        other.numel_ = 0;
        other.nbytes_ = 0;
    }
    return *this;
}

Tensor::~Tensor() { release_(); }

void Tensor::compute_default_stride_() {
    stride_.assign(shape_.size(), 0);
    int64_t acc = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        stride_[i] = acc;
        if (shape_[i] > 0) acc *= shape_[i];
    }
}

void Tensor::allocate_() {
    if (nbytes_ == 0) {
        data_ = nullptr;
        owns_data_ = false;
        return;
    }
    if (device_.is_cuda()) {
        // Use cudaMalloc for now; the bump Allocator will be plugged in
        // by callers that want a unified memory pool (Week 5+).
        void* ptr = nullptr;
        MI_CUDA_CHECK(cudaMalloc(&ptr, nbytes_));
        data_ = ptr;
    } else {
        // 256B alignment (cuBLAS requirement) keeps host staging buffers
        // compatible with future SIMD / device-side pinned transfers.
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 256, nbytes_) != 0) {
            throw std::bad_alloc();
        }
        data_ = ptr;
    }
    owns_data_ = true;
}

void Tensor::release_() {
    if (!owns_data_ || data_ == nullptr) {
        data_ = nullptr;
        owns_data_ = false;
        return;
    }
    if (device_.is_cuda()) {
        // Ignore errors during destructor; we're tearing down anyway.
        cudaFree(data_);
    } else {
        std::free(data_);
    }
    data_ = nullptr;
    owns_data_ = false;
}

bool Tensor::is_contiguous() const {
    if (static_cast<int>(stride_.size()) != static_cast<int>(shape_.size())) {
        return false;
    }
    int64_t expected = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (shape_[i] == 1) continue;  // broadcast dim
        if (stride_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

void Tensor::copy_raw(const void* src, void* dst, std::size_t bytes,
                      Device src_dev, Device dst_dev) {
    if (bytes == 0) return;
    if (src_dev.is_cuda() && dst_dev.is_cuda()) {
        MI_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    } else if (!src_dev.is_cuda() && !dst_dev.is_cuda()) {
        std::memcpy(dst, src, bytes);
    } else if (!src_dev.is_cuda() && dst_dev.is_cuda()) {
        MI_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    } else {  // src CUDA, dst CPU
        MI_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    }
}

Tensor Tensor::to(Device target) const {
    Tensor out(shape_, dtype_, target);
    if (nbytes_ == 0) return out;
    copy_raw(data_, out.data_, nbytes_, device_, target);
    return out;
}

void Tensor::fill(float value) {
    if (nbytes_ == 0 || data_ == nullptr) return;
    if (device_.is_cpu()) {
        if (dtype_ == DType::FP32) {
            for (int64_t i = 0; i < numel_; ++i) {
                static_cast<float*>(data_)[i] = value;
            }
        } else if (dtype_ == DType::FP16) {
            const uint16_t bits = f32_to_f16_bits(value);
            auto* p = static_cast<uint16_t*>(data_);
            for (int64_t i = 0; i < numel_; ++i) p[i] = bits;
        } else {
            throw std::runtime_error("Tensor::fill CPU: dtype not supported yet");
        }
    } else {
        // CUDA device
        if (value == 0.0f) {
            MI_CUDA_CHECK(cudaMemset(data_, 0, nbytes_));
            return;
        }
        throw std::runtime_error(
            "Tensor::fill on CUDA for non-zero value is not implemented in "
            "Week 1; use host staging + h2d or an elementwise kernel "
            "(will land in Week 2).");
    }
}

}  // namespace mini_infer