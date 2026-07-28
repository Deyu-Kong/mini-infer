#include "layers/mlp.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "kernels/gelu_kernel.cuh"
#include "kernels/model_utils_kernel.cuh"
#include "kernels/swiglu_kernel.cuh"

namespace mini_infer {

// Debug helper: print tensor statistics
static void debug_print_tensor(const char* name, const Tensor& t, int max_elements = 10) {
    if (t.dtype() != DType::FP16) {
        std::printf("[DEBUG] %s: not FP16, skipping\n", name);
        return;
    }
    
    // Copy to CPU
    Tensor cpu_t = t.to(Device::cpu());
    const __half* data = static_cast<const __half*>(cpu_t.data());
    int64_t numel = cpu_t.numel();
    
    // Compute mean and std
    double sum = 0.0, sum_sq = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
        float val = __half2float(data[i]);
        sum += val;
        sum_sq += val * val;
    }
    double mean = sum / numel;
    double variance = (sum_sq / numel) - (mean * mean);
    double std = std::sqrt(variance);
    
    std::printf("[DEBUG] %s: shape=[", name);
    for (size_t i = 0; i < t.shape().size(); ++i) {
        if (i > 0) std::printf(",");
        std::printf("%ld", t.shape()[i]);
    }
    std::printf("], mean=%.6f, std=%.6f", mean, std);
    
    // Print first few elements
    if (max_elements > 0 && numel > 0) {
        std::printf(", first_%ld=[", std::min((int64_t)max_elements, numel));
        for (int64_t i = 0; i < std::min((int64_t)max_elements, numel); ++i) {
            if (i > 0) std::printf(",");
            std::printf("%.6f", __half2float(data[i]));
        }
        std::printf("]");
    }
    std::printf("\n");
}

namespace {
inline cublasHandle_t get_cublas(void* h) {
    return reinterpret_cast<cublasHandle_t>(h);
}

void cublas_check_(cublasStatus_t s, const char* expr, const char* file, int line) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error ") + expr + " at " +
                                 file + ":" + std::to_string(line));
    }
}
#define MI_CHECK_CUBLAS(expr) cublas_check_((expr), #expr, __FILE__, __LINE__)

void cuda_check_(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error ") + expr + " at " +
                                 file + ":" + std::to_string(line) + " : " +
                                 cudaGetErrorString(e));
    }
}
#define MI_CHECK_CUDA(expr) cuda_check_((expr), #expr, __FILE__, __LINE__)

/**
 * Row-major C[M,N] = A[M,K] @ B[K,N]   with FP16 inputs and FP32 accumulate.
 *
 * cuBLAS is column-major. We exploit the identity:
 *
 *   row_major(X)[i, j]  has the same memory as col_major(X^T)[i, j]
 *   but  row_major(X)[i, j]  ==  col_major(X^T)[j, i].
 *
 * i.e. cuBLAS sees the data transposed. So if we want cuBLAS to operate on
 * our row-major A as a true (M, K) matrix, we hand it the data and use
 * OP_T to flip it back.
 *
 * Derivation (working through col-major algebra):
 *
 *   row-major C = row-major A @ row-major B
 *                = (col A_view)^T @ (col B_view)^T
 *
 *   col C_view  = ( (col A_view)^T @ (col B_view)^T )^T
 *               = (col B_view) @ (col A_view)^T
 *
 * so cuBLAS computes
 *
 *   cuBLAS C   = (col B_view) @ (col A_view)^T
 *              = OP_N on B-view  *  OP_T on A-view
 *
 * with shapes  (N x M)  =  (N x K)  *  (K x M)
 *
 *   cuBLAS A arg = our row-major B  (col-major shape (K, N), leading dim K)
 *   cuBLAS B arg = our row-major A  (col-major shape (K, M), leading dim K)
 *   cuBLAS C arg = our row-major C  (col-major shape (N, M), leading dim N)
 */
void gemm_rowmajor_f16(cublasHandle_t handle,
                       int M, int N, int K,
                       const __half* A, int lda,   // A is [M, K]  row-major
                       const __half* B, int ldb,   // B is [K, N]  row-major
                       __half* C, int ldc,         // C is [M, N]  row-major
                       float alpha, float beta) {
    (void)lda; (void)ldb; (void)ldc;
    MI_CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, K,    // cuBLAS A = row-major B (col-major shape (K, N)), lda=K
        A, CUDA_R_16F, K,    // cuBLAS B = row-major A (col-major shape (K, M)), ldb=K
        &beta,
        C, CUDA_R_16F, N,    // cuBLAS C = row-major C (col-major shape (N, M)), ldc=N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}
}  // namespace

MLP::MLP(int64_t hidden, int64_t intermediate, int device_index, ActKind act)
    : hidden_(hidden), intermediate_(intermediate), device_index_(device_index), act_(act),
      w_gate_(Tensor::empty({intermediate, hidden}, DType::FP16, Device::cuda(device_index))),
      w_up_(Tensor::empty({intermediate, hidden}, DType::FP16, Device::cuda(device_index))),
      w_down_(Tensor::empty({hidden, intermediate}, DType::FP16, Device::cuda(device_index))) {
    MI_CHECK_CUDA(cudaSetDevice(device_index_));
    cublasStatus_t s = cublasCreate(reinterpret_cast<cublasHandle_t*>(&cublas_handle_));
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed");
    }
    // Match math mode to compute type for reproducibility.
    MI_CHECK_CUBLAS(cublasSetMathMode(get_cublas(cublas_handle_),
                                      CUBLAS_DEFAULT_MATH));
}

MLP::~MLP() {
    if (cublas_handle_) {
        cublasDestroy(get_cublas(cublas_handle_));
        cublas_handle_ = nullptr;
    }
}

MLP::MLP(MLP&& other) noexcept {
    hidden_       = other.hidden_;
    intermediate_ = other.intermediate_;
    device_index_ = other.device_index_;
    act_          = other.act_;
    w_gate_       = std::move(other.w_gate_);
    w_up_         = std::move(other.w_up_);
    w_down_       = std::move(other.w_down_);
    gate_buf_     = std::move(other.gate_buf_);
    up_buf_       = std::move(other.up_buf_);
    act_buf_      = std::move(other.act_buf_);
    cublas_handle_ = other.cublas_handle_;
    other.cublas_handle_ = nullptr;
}

MLP& MLP::operator=(MLP&& other) noexcept {
    if (this != &other) {
        if (cublas_handle_) cublasDestroy(get_cublas(cublas_handle_));
        hidden_       = other.hidden_;
        intermediate_ = other.intermediate_;
        device_index_ = other.device_index_;
        act_          = other.act_;
        w_gate_       = std::move(other.w_gate_);
        w_up_         = std::move(other.w_up_);
        w_down_       = std::move(other.w_down_);
        gate_buf_     = std::move(other.gate_buf_);
        up_buf_       = std::move(other.up_buf_);
        act_buf_      = std::move(other.act_buf_);
        cublas_handle_ = other.cublas_handle_;
        other.cublas_handle_ = nullptr;
    }
    return *this;
}

void MLP::init(int64_t hidden, int64_t intermediate, int device_index, ActKind act) {
    if (hidden_ != 0) return;  // already initialized
    hidden_ = hidden;
    intermediate_ = intermediate;
    device_index_ = device_index;
    act_ = act;
    // Don't allocate weights here - they will be set by set_weights()
    MI_CHECK_CUDA(cudaSetDevice(device_index));
    cublasStatus_t s = cublasCreate(reinterpret_cast<cublasHandle_t*>(&cublas_handle_));
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed");
    }
    MI_CHECK_CUBLAS(cublasSetMathMode(get_cublas(cublas_handle_),
                                      CUBLAS_DEFAULT_MATH));
}

void MLP::set_weights(const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down) {
    auto need = [&](const Tensor& t, const std::vector<int64_t>& sh) {
        if (t.dtype() != DType::FP16) {
            throw std::runtime_error("MLP weights must be FP16");
        }
        if (t.shape() != sh) {
            std::fprintf(stderr, "Shape mismatch: got [%ld,%ld] expected [%ld,%ld]\n",
                         t.shape()[0], t.shape()[1], sh[0], sh[1]);
            throw std::runtime_error("MLP weight shape mismatch");
        }
    };
    need(w_gate, {intermediate_, hidden_});
    need(w_up,   {intermediate_, hidden_});
    need(w_down, {hidden_, intermediate_});
    w_gate_ = w_gate.to(Device::cuda(device_index_));
    w_up_   = w_up.to(Device::cuda(device_index_));
    w_down_ = w_down.to(Device::cuda(device_index_));
}

Tensor MLP::forward(const Tensor& x) {
    if (x.dtype() != DType::FP16) {
        throw std::runtime_error("MLP::forward expects FP16");
    }
    if (x.ndim() != 2) {
        throw std::runtime_error("MLP::forward expects 2-D input");
    }
    if (x.shape()[1] != hidden_) {
        throw std::runtime_error("MLP::forward: hidden dim mismatch");
    }
    if (!x.is_contiguous()) {
        throw std::runtime_error("MLP::forward expects contiguous input");
    }

    const int B = static_cast<int>(x.shape()[0]);

    // Lazily allocate intermediates sized for the current B.
    if (gate_buf_.numel() != static_cast<int64_t>(B) * intermediate_) {
        gate_buf_ = Tensor::empty({B, intermediate_}, DType::FP16, Device::cuda(device_index_));
        up_buf_   = Tensor::empty({B, intermediate_}, DType::FP16, Device::cuda(device_index_));
        act_buf_  = Tensor::empty({B, intermediate_}, DType::FP16, Device::cuda(device_index_));
    }

    Tensor y = Tensor::empty({B, hidden_}, DType::FP16, Device::cuda(device_index_));

    auto handle = get_cublas(cublas_handle_);
    const float one = 1.0f;
    const float zero = 0.0f;

    // gate = x @ W_gate^T   (M=B, N=I, K=H)
    gemm_rowmajor_f16(handle,
        B, static_cast<int>(intermediate_), static_cast<int>(hidden_),
        static_cast<const __half*>(x.data()), static_cast<int>(hidden_),
        static_cast<const __half*>(w_gate_.data()), static_cast<int>(hidden_),
        static_cast<__half*>(gate_buf_.data()), static_cast<int>(intermediate_),
        one, zero);

    // up   = x @ W_up^T
    gemm_rowmajor_f16(handle,
        B, static_cast<int>(intermediate_), static_cast<int>(hidden_),
        static_cast<const __half*>(x.data()), static_cast<int>(hidden_),
        static_cast<const __half*>(w_up_.data()), static_cast<int>(hidden_),
        static_cast<__half*>(up_buf_.data()), static_cast<int>(intermediate_),
        one, zero);

// Activation * up. Two kernels:
    //   - SwiGLU  : fused silu(gate) * up   (one elementwise kernel)
    //   - GeGLU   : gelu_tanh(gate), then in-place multiply by up
    if (act_ == ActKind::Silu) {
        kernels::launch_swiglu(
            static_cast<const __half*>(gate_buf_.data()),
            static_cast<const __half*>(up_buf_.data()),
            static_cast<__half*>(act_buf_.data()),
            static_cast<int>(B * intermediate_),
            /*stream=*/0);
    } else {
        // GeGLU: act_buf = gelu_tanh(gate); then act_buf *= up
        kernels::launch_gelu_tanh(
            static_cast<const __half*>(gate_buf_.data()),
            static_cast<__half*>(act_buf_.data()),
            static_cast<int64_t>(B) * intermediate_,
            /*stream=*/0);
        kernels::launch_mul_inplace(
            static_cast<__half*>(act_buf_.data()),
            static_cast<const __half*>(up_buf_.data()),
            static_cast<int64_t>(B) * intermediate_,
            /*stream=*/0);
    }

    // down = act_buf @ W_down^T  (M=B, N=H, K=I)
    gemm_rowmajor_f16(handle,
        B, static_cast<int>(hidden_), static_cast<int>(intermediate_),
        static_cast<const __half*>(act_buf_.data()), static_cast<int>(intermediate_),
        static_cast<const __half*>(w_down_.data()), static_cast<int>(intermediate_),
        static_cast<__half*>(y.data()), static_cast<int>(hidden_),
        one, zero);

    return y;
}

}  // namespace mini_infer