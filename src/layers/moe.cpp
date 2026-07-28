#include "layers/moe.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "kernels/moe_kernel.cuh"

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

void gemm_rowmajor_f16(cublasHandle_t handle,
                       int M, int N, int K,
                       const __half* A, int lda,
                       const __half* B, int ldb,
                       __half* C, int ldc,
                       float alpha, float beta) {
    (void)lda; (void)ldb; (void)ldc;
    MI_CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, K,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}
}  // namespace

MoELayer::MoELayer(int64_t hidden, int64_t moe_intermediate, int64_t num_experts,
                   int64_t num_experts_per_tok, int device_index, ActKind act)
    : hidden_(hidden),
      moe_intermediate_(moe_intermediate),
      num_experts_(num_experts),
      num_experts_per_tok_(num_experts_per_tok),
      device_index_(device_index),
      act_(act),
      w_router_gate_(Tensor::empty({num_experts, hidden}, DType::FP16,
                                    Device::cuda(device_index))) {
    MI_CHECK_CUDA(cudaSetDevice(device_index_));
    cublasStatus_t s = cublasCreate(reinterpret_cast<cublasHandle_t*>(&cublas_handle_));
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed");
    }
    MI_CHECK_CUBLAS(cublasSetMathMode(get_cublas(cublas_handle_),
                                       CUBLAS_DEFAULT_MATH));

    experts_.reserve(num_experts_);
    for (int64_t i = 0; i < num_experts_; ++i) {
        experts_.emplace_back(hidden, moe_intermediate, device_index, act);
    }
}

MoELayer::~MoELayer() {
    if (cublas_handle_) {
        cublasDestroy(get_cublas(cublas_handle_));
        cublas_handle_ = nullptr;
    }
}

MoELayer::MoELayer(MoELayer&& other) noexcept {
    hidden_ = other.hidden_;
    moe_intermediate_ = other.moe_intermediate_;
    num_experts_ = other.num_experts_;
    num_experts_per_tok_ = other.num_experts_per_tok_;
    device_index_ = other.device_index_;
    act_ = other.act_;
    w_router_gate_ = std::move(other.w_router_gate_);
    experts_ = std::move(other.experts_);
    router_logits_buf_ = std::move(other.router_logits_buf_);
    expert_weights_buf_ = std::move(other.expert_weights_buf_);
    expert_indices_buf_ = std::move(other.expert_indices_buf_);
    cublas_handle_ = other.cublas_handle_;
    other.cublas_handle_ = nullptr;
}

MoELayer& MoELayer::operator=(MoELayer&& other) noexcept {
    if (this != &other) {
        if (cublas_handle_) cublasDestroy(get_cublas(cublas_handle_));
        hidden_ = other.hidden_;
        moe_intermediate_ = other.moe_intermediate_;
        num_experts_ = other.num_experts_;
        num_experts_per_tok_ = other.num_experts_per_tok_;
        device_index_ = other.device_index_;
        act_ = other.act_;
        w_router_gate_ = std::move(other.w_router_gate_);
        experts_ = std::move(other.experts_);
        router_logits_buf_ = std::move(other.router_logits_buf_);
        expert_weights_buf_ = std::move(other.expert_weights_buf_);
        expert_indices_buf_ = std::move(other.expert_indices_buf_);
        cublas_handle_ = other.cublas_handle_;
        other.cublas_handle_ = nullptr;
    }
    return *this;
}

void MoELayer::init(int64_t hidden, int64_t moe_intermediate, int64_t num_experts,
                    int64_t num_experts_per_tok, int device_index, ActKind act) {
    if (hidden_ != 0) return;
    hidden_ = hidden;
    moe_intermediate_ = moe_intermediate;
    num_experts_ = num_experts;
    num_experts_per_tok_ = num_experts_per_tok;
    device_index_ = device_index;
    act_ = act;
    MI_CHECK_CUDA(cudaSetDevice(device_index));
    cublasStatus_t s = cublasCreate(reinterpret_cast<cublasHandle_t*>(&cublas_handle_));
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed");
    }
    MI_CHECK_CUBLAS(cublasSetMathMode(get_cublas(cublas_handle_),
                                       CUBLAS_DEFAULT_MATH));
    experts_.reserve(num_experts_);
    for (int64_t i = 0; i < num_experts_; ++i) {
        experts_.emplace_back();
        experts_.back().init(hidden, moe_intermediate, device_index, act);
    }
}

void MoELayer::set_router_gate(const Tensor& w_gate) {
    if (w_gate.shape() != std::vector<int64_t>{num_experts_, hidden_}) {
        throw std::runtime_error("MoELayer: router gate shape mismatch");
    }
    w_router_gate_ = w_gate.to(Device::cuda(device_index_));
}

void MoELayer::set_expert_weights(int64_t expert_idx, const Tensor& w_gate,
                                  const Tensor& w_up, const Tensor& w_down) {
    if (expert_idx < 0 || expert_idx >= num_experts_) {
        throw std::runtime_error("MoELayer: expert index out of range");
    }
    experts_[expert_idx].set_weights(w_gate, w_up, w_down);
}

Tensor MoELayer::forward(const Tensor& x) {
    if (x.dtype() != DType::FP16) {
        throw std::runtime_error("MoELayer::forward expects FP16");
    }
    if (x.ndim() != 2) {
        throw std::runtime_error("MoELayer::forward expects 2-D input");
    }
    if (x.shape()[1] != hidden_) {
        throw std::runtime_error("MoELayer::forward: hidden dim mismatch");
    }

    const int B = static_cast<int>(x.shape()[0]);
    const int E = static_cast<int>(num_experts_);
    const int K = static_cast<int>(num_experts_per_tok_);

    // Lazily allocate buffers
    if (router_logits_buf_.numel() != static_cast<int64_t>(B) * E) {
        router_logits_buf_ = Tensor::empty({B, E}, DType::FP16,
                                            Device::cuda(device_index_));
    }
    if (expert_weights_buf_.numel() != static_cast<int64_t>(B) * K) {
        expert_weights_buf_ = Tensor::empty({B, K}, DType::FP32,
                                             Device::cuda(device_index_));
        expert_indices_buf_ = Tensor::empty({B, K}, DType::INT32,
                                             Device::cuda(device_index_));
    }

    // 1. Router gate: logits = x @ W_gate^T  -> [B, E]
    auto handle = get_cublas(cublas_handle_);
    const float one = 1.0f, zero = 0.0f;
    gemm_rowmajor_f16(handle,
        B, E, static_cast<int>(hidden_),
        static_cast<const __half*>(x.data()), static_cast<int>(hidden_),
        static_cast<const __half*>(w_router_gate_.data()), static_cast<int>(hidden_),
        static_cast<__half*>(router_logits_buf_.data()), E,
        one, zero);

    // 2. Top-K selection + softmax weighting
    kernels::launch_moe_topk_select(
        static_cast<const __half*>(router_logits_buf_.data()),
        static_cast<float*>(expert_weights_buf_.data()),
        static_cast<int*>(expert_indices_buf_.data()),
        B, E, K, /*stream=*/0);

    // 3. Run each expert and accumulate weighted outputs
    // Strategy: for each expert, run it on all tokens, then use scatter-add
    // to accumulate only for tokens that selected this expert.
    // This is simpler than gather-scatter but runs all experts on all tokens.
    Tensor output = Tensor::empty({B, hidden_}, DType::FP16,
                                   Device::cuda(device_index_));
    MI_CHECK_CUDA(cudaMemset(output.data(), 0,
                              static_cast<int64_t>(B) * hidden_ * sizeof(__half)));

    // Copy expert_indices to host for routing
    std::vector<int> expert_indices_h(B * K);
    MI_CHECK_CUDA(cudaMemcpy(expert_indices_h.data(),
                              expert_indices_buf_.data(),
                              B * K * sizeof(int),
                              cudaMemcpyDeviceToHost));

    // For each expert, check if any token selected it, run if needed
    for (int e = 0; e < E; ++e) {
        bool expert_used = false;
        for (int b = 0; b < B && !expert_used; ++b) {
            for (int k = 0; k < K; ++k) {
                if (expert_indices_h[b * K + k] == e) {
                    expert_used = true;
                    break;
                }
            }
        }
        if (!expert_used) continue;

        // Run this expert on all tokens (could optimize to only selected tokens)
        Tensor expert_out = experts_[e].forward(x);

        // Accumulate for each slot k where this expert was selected
        for (int k = 0; k < K; ++k) {
            kernels::launch_moe_scatter_add(
                static_cast<__half*>(output.data()),
                static_cast<const __half*>(expert_out.data()),
                static_cast<const float*>(expert_weights_buf_.data()),
                static_cast<const int*>(expert_indices_buf_.data()),
                k, e, B, static_cast<int>(hidden_), K, /*stream=*/0);
        }
    }

    return output;
}

}  // namespace mini_infer
