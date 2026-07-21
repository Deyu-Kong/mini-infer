#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mini_infer {

enum class DType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT32 = 3,
    INT64 = 4,
};

enum class DeviceType : uint8_t {
    CPU = 0,
    CUDA = 1,
};

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;  // GPU id for CUDA; ignored for CPU

    bool is_cuda() const { return type == DeviceType::CUDA; }
    bool is_cpu() const { return type == DeviceType::CPU; }

    static Device cpu() { return {DeviceType::CPU, 0}; }
    static Device cuda(int idx = 0) { return {DeviceType::CUDA, idx}; }
};

inline std::size_t dtype_size(DType d) {
    switch (d) {
        case DType::FP32:  return 4;
        case DType::FP16:  return 2;
        case DType::BF16:  return 2;
        case DType::INT32: return 4;
        case DType::INT64: return 8;
    }
    return 0;
}

const char* dtype_name(DType d);

/**
 * Tensor — lightweight n-dim array descriptor.
 *
 * Owns a contiguous block of `nbytes_` bytes when `owns_data_` is true.
 * Holds shape + stride (in elements) so views into the same memory can
 * be expressed without copying (will be heavily used by PagedAttention).
 *
 * Week-1 scope:
 *   - shape / stride / dtype / device
 *   - host<->device copy
 *   - contiguous (row-major) creation
 *   - simple fill / element-wise accessor helpers
 */
class Tensor {
public:
    Tensor();
    Tensor(std::vector<int64_t> shape, DType dtype, Device device);

    // Factory: allocates and owns memory.
    static Tensor empty(std::vector<int64_t> shape, DType dtype, Device device);

    // Rule of five: copy is disallowed (would double-free), move is allowed.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    // ---------- accessors ----------
    const std::vector<int64_t>& shape()  const { return shape_; }
    const std::vector<int64_t>& stride() const { return stride_; }
    int64_t numel() const { return numel_; }
    std::size_t nbytes() const { return nbytes_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    void* data() { return data_; }
    const void* data() const { return data_; }
    bool owns_data() const { return owns_data_; }

    // ---------- transfer ----------
    // Copy to a target device (CPU<->CUDA). Returns a new tensor.
    Tensor to(Device target) const;

    // Copy raw bytes between two device pointers of equal size.
    static void copy_raw(const void* src, void* dst, std::size_t bytes,
                         Device src_dev, Device dst_dev);

    // Fill with a constant (host scalar, broadcast to all elements).
    void fill(float value);

    // Stride helpers
    bool is_contiguous() const;

private:
    void allocate_();
    void release_();
    void compute_default_stride_();

    std::vector<int64_t> shape_;
    std::vector<int64_t> stride_;
    int64_t numel_ = 0;
    std::size_t nbytes_ = 0;
    DType dtype_ = DType::FP32;
    Device device_ = Device::cpu();
    void* data_ = nullptr;
    bool owns_data_ = false;
};

}  // namespace mini_infer