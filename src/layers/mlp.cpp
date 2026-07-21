#include "layers/mlp.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/swiglu_kernel.cuh"

namespace mini_infer {

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

MLP::MLP(int64_t hidden, int64_t intermediate, int device_index)
    : hidden_(hidden), intermediate_(intermediate), device_index_(device_index),
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

void MLP::set_weights(const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down) {
    auto need = [&](const Tensor& t, const std::vector<int64_t>& sh) {
        if (t.dtype() != DType::FP16) {
            throw std::runtime_error("MLP weights must be FP16");
        }
        if (t.shape() != sh) {
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
        silu_buf_ = Tensor::empty({B, intermediate_}, DType::FP16, Device::cuda(device_index_));
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

    // SwiGLU: silu(gate) * up — fused kernel reads gate, up, writes silu_buf.
    kernels::launch_swiglu(
        static_cast<const __half*>(gate_buf_.data()),
        static_cast<const __half*>(up_buf_.data()),
        static_cast<__half*>(silu_buf_.data()),
        static_cast<int>(B * intermediate_),
        /*stream=*/0);

    // down = silu_buf @ W_down^T  (M=B, N=H, K=I)
    gemm_rowmajor_f16(handle,
        B, static_cast<int>(hidden_), static_cast<int>(intermediate_),
        static_cast<const __half*>(silu_buf_.data()), static_cast<int>(intermediate_),
        static_cast<const __half*>(w_down_.data()), static_cast<int>(intermediate_),
        static_cast<__half*>(y.data()), static_cast<int>(hidden_),
        one, zero);

    return y;
}

}  // namespace mini_infer