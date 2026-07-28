#include "layers/attention.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "kernels/model_utils_kernel.cuh"
#include "kernels/paged_attn_kernel.cuh"
#include "kernels/rmsnorm_kernel.cuh"
#include "scheduler/paged_kv_cache.h"

namespace mini_infer {

namespace {
inline cublasHandle_t get_cublas(void* h) { return reinterpret_cast<cublasHandle_t>(h); }
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

// row-major C[M,N] = A[M,K] @ B[K,N], FP16 in, FP32 accumulate.
void gemm_rowmajor(cublasHandle_t handle,
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

Attention::Attention(int64_t hidden, int64_t num_heads, int64_t num_kv_heads,
                     int64_t head_dim, float rope_theta, int device_index)
    : hidden_(hidden), num_heads_(num_heads), num_kv_heads_(num_kv_heads),
      head_dim_(head_dim), rope_theta_(rope_theta), device_index_(device_index) {
    init(hidden, num_heads, num_kv_heads, head_dim, rope_theta, device_index);
}

void Attention::init(int64_t hidden, int64_t num_heads, int64_t num_kv_heads,
                     int64_t head_dim, float rope_theta, int device_index) {
    hidden_ = hidden;
    num_heads_ = num_heads;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    rope_theta_ = rope_theta;
    device_index_ = device_index;
    
    if (hidden % num_heads != 0) {
        throw std::runtime_error("Attention: hidden % num_heads != 0");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("Attention: num_heads % num_kv_heads != 0 (GQA)");
    }

    MI_CHECK_CUDA(cudaSetDevice(device_index_));
    MI_CHECK_CUBLAS(cublasCreate(reinterpret_cast<cublasHandle_t*>(&cublas_handle_)));
    MI_CHECK_CUBLAS(cublasSetMathMode(get_cublas(cublas_handle_),
                                      CUBLAS_DEFAULT_MATH));

    // RoPE uses half_dim = head_dim/2. We share inv_freq between Q and K.
    rope_q_ = std::make_unique<RoPE>(head_dim, rope_theta_, device_index_);
    rope_k_ = std::make_unique<RoPE>(head_dim, rope_theta_, device_index_);
}

void Attention::set_qk_norm(int64_t head_dim, float eps, int device_index) {
    use_qk_norm_ = true;
    qk_norm_eps_ = eps;
    // Allocate [head_dim] FP16 buffers; they get filled by set_qk_norm_weights.
    w_q_norm_ = Tensor::empty({head_dim}, DType::FP16, Device::cuda(device_index));
    w_k_norm_ = Tensor::empty({head_dim}, DType::FP16, Device::cuda(device_index));
}

Attention::~Attention() {
    if (cublas_handle_) {
        cublasDestroy(get_cublas(cublas_handle_));
        cublas_handle_ = nullptr;
    }
}

Attention::Attention(Attention&& other) noexcept
    : hidden_(other.hidden_),
      num_heads_(other.num_heads_),
      num_kv_heads_(other.num_kv_heads_),
      head_dim_(other.head_dim_),
      rope_theta_(other.rope_theta_),
      device_index_(other.device_index_),
      w_q_(std::move(other.w_q_)),
      w_k_(std::move(other.w_k_)),
      w_v_(std::move(other.w_v_)),
      w_o_(std::move(other.w_o_)),
      b_q_(std::move(other.b_q_)),
      b_k_(std::move(other.b_k_)),
      b_v_(std::move(other.b_v_)),
      has_bias_(other.has_bias_),
      w_q_norm_(std::move(other.w_q_norm_)),
      w_k_norm_(std::move(other.w_k_norm_)),
      use_qk_norm_(other.use_qk_norm_),
      qk_norm_eps_(other.qk_norm_eps_),
      sliding_window_(other.sliding_window_),
      use_local_rope_(other.use_local_rope_),
      local_rope_theta_(other.local_rope_theta_),
      q_buf_(std::move(other.q_buf_)),
      k_buf_(std::move(other.k_buf_)),
      v_buf_(std::move(other.v_buf_)),
      attn_out_buf_(std::move(other.attn_out_buf_)),
      k_scratch_(std::move(other.k_scratch_)),
      v_scratch_(std::move(other.v_scratch_)),
      rope_q_(std::move(other.rope_q_)),
      rope_k_(std::move(other.rope_k_)),
      cublas_handle_(other.cublas_handle_) {
    other.cublas_handle_ = nullptr;
}

Attention& Attention::operator=(Attention&& other) noexcept {
    if (this != &other) {
        if (cublas_handle_) {
            cublasDestroy(get_cublas(cublas_handle_));
        }
        hidden_ = other.hidden_;
        num_heads_ = other.num_heads_;
        num_kv_heads_ = other.num_kv_heads_;
        head_dim_ = other.head_dim_;
        rope_theta_ = other.rope_theta_;
        device_index_ = other.device_index_;
        w_q_ = std::move(other.w_q_);
        w_k_ = std::move(other.w_k_);
        w_v_ = std::move(other.w_v_);
        w_o_ = std::move(other.w_o_);
        b_q_ = std::move(other.b_q_);
        b_k_ = std::move(other.b_k_);
        b_v_ = std::move(other.b_v_);
        has_bias_ = other.has_bias_;
        w_q_norm_ = std::move(other.w_q_norm_);
        w_k_norm_ = std::move(other.w_k_norm_);
        use_qk_norm_ = other.use_qk_norm_;
        qk_norm_eps_ = other.qk_norm_eps_;
        sliding_window_ = other.sliding_window_;
        use_local_rope_ = other.use_local_rope_;
        local_rope_theta_ = other.local_rope_theta_;
        q_buf_ = std::move(other.q_buf_);
        k_buf_ = std::move(other.k_buf_);
        v_buf_ = std::move(other.v_buf_);
        attn_out_buf_ = std::move(other.attn_out_buf_);
        k_scratch_ = std::move(other.k_scratch_);
        v_scratch_ = std::move(other.v_scratch_);
        rope_q_ = std::move(other.rope_q_);
        rope_k_ = std::move(other.rope_k_);
        cublas_handle_ = other.cublas_handle_;
        other.cublas_handle_ = nullptr;
    }
    return *this;
}

void Attention::set_weights(Tensor w_q, Tensor w_k, Tensor w_v, Tensor w_o,
                            Tensor b_q, Tensor b_k, Tensor b_v) {
    auto need = [&](const Tensor& t, const std::vector<int64_t>& sh,
                    const std::string& n) {
        if (t.dtype() != DType::FP16) {
            throw std::runtime_error("Attention weight " + n + " must be FP16");
        }
        if (t.shape() != sh) {
            throw std::runtime_error("Attention weight " + n + " shape mismatch");
        }
    };
    need(w_q, {num_heads_    * head_dim_, hidden_}, "W_q");
    need(w_k, {num_kv_heads_ * head_dim_, hidden_}, "W_k");
    need(w_v, {num_kv_heads_ * head_dim_, hidden_}, "W_v");
    need(w_o, {hidden_, num_heads_    * head_dim_}, "W_o");
    w_q_ = w_q.to(Device::cuda(device_index_));
    w_k_ = w_k.to(Device::cuda(device_index_));
    w_v_ = w_v.to(Device::cuda(device_index_));
    w_o_ = w_o.to(Device::cuda(device_index_));

    has_bias_ = (b_q.numel() > 0);
    if (has_bias_) {
        need(b_q, {num_heads_    * head_dim_}, "b_q");
        need(b_k, {num_kv_heads_ * head_dim_}, "b_k");
        need(b_v, {num_kv_heads_ * head_dim_}, "b_v");
        b_q_ = b_q.to(Device::cuda(device_index_));
        b_k_ = b_k.to(Device::cuda(device_index_));
        b_v_ = b_v.to(Device::cuda(device_index_));
    }
}

void Attention::set_qk_norm_weights(const Tensor& w_q_norm,
                                   const Tensor& w_k_norm) {
    if (!use_qk_norm_) {
        throw std::runtime_error(
            "Attention::set_qk_norm_weights: call set_qk_norm() first");
    }
    if (w_q_norm.dtype() != DType::FP16 || w_k_norm.dtype() != DType::FP16) {
        throw std::runtime_error("Attention qk_norm weights must be FP16");
    }
    if (w_q_norm.numel() != head_dim_ || w_k_norm.numel() != head_dim_) {
        throw std::runtime_error("Attention qk_norm weight size mismatch");
    }
    w_q_norm_ = w_q_norm.to(Device::cuda(device_index_));
    w_k_norm_ = w_k_norm.to(Device::cuda(device_index_));
}
Tensor Attention::forward(const Tensor& hidden_states,
                          const std::vector<int64_t>& positions,
                          __half* kv_k_ptr,
                          __half* kv_v_ptr,
                          int64_t max_seq,
                          int64_t cur_len,
                          bool is_prefill) {
    if (hidden_states.dtype() != DType::FP16) {
        throw std::runtime_error("Attention::forward expects FP16 input");
    }
    if (hidden_states.ndim() != 3) {
        throw std::runtime_error("Attention::forward expects 3-D input [B,S,H]");
    }
    const int B = static_cast<int>(hidden_states.shape()[0]);
    const int S = static_cast<int>(hidden_states.shape()[1]);
    const int H = static_cast<int>(hidden_);
    const int Hq = static_cast<int>(num_heads_    * head_dim_);
    const int Hkv = static_cast<int>(num_kv_heads_ * head_dim_);

    // Lazily allocate persistent buffers.
    auto ensure = [&](Tensor& buf, const std::vector<int64_t>& sh, DType dt) {
        if (buf.numel() != sh[0] * sh[1] * sh[2]) {
            buf = Tensor::empty(sh, dt, Device::cuda(device_index_));
        }
    };
    ensure(q_buf_,         {B, S, Hq},  DType::FP16);
    ensure(k_scratch_,     {B, S, Hkv}, DType::FP16);
    ensure(v_scratch_,     {B, S, Hkv}, DType::FP16);
    ensure(attn_out_buf_,  {B, S, Hq},  DType::FP16);

    auto handle = get_cublas(cublas_handle_);
    const float one  = 1.0f;
    const float zero = 0.0f;

    // Q = x @ W_q^T  (M=B*S, N=Hq, K=H)
    {
        const int Mflat = B * S;
        gemm_rowmajor(handle, Mflat, Hq, H,
            static_cast<const __half*>(hidden_states.data()), H,
            static_cast<const __half*>(w_q_.data()), H,
            static_cast<__half*>(q_buf_.data()), Hq,
            one, zero);
        if (has_bias_) {
            kernels::launch_add_bias(static_cast<__half*>(q_buf_.data()),
                            static_cast<const __half*>(b_q_.data()),
                            Mflat, Hq, Mflat * Hq, /*stream=*/0);
        }
    }
    // K, V
    {
        const int Mflat = B * S;
        gemm_rowmajor(handle, Mflat, Hkv, H,
            static_cast<const __half*>(hidden_states.data()), H,
            static_cast<const __half*>(w_k_.data()), H,
            static_cast<__half*>(k_scratch_.data()), Hkv,
            one, zero);
        if (has_bias_) {
            kernels::launch_add_bias(static_cast<__half*>(k_scratch_.data()),
                            static_cast<const __half*>(b_k_.data()),
                            Mflat, Hkv, Mflat * Hkv, /*stream=*/0);
        }
        gemm_rowmajor(handle, Mflat, Hkv, H,
            static_cast<const __half*>(hidden_states.data()), H,
            static_cast<const __half*>(w_v_.data()), H,
            static_cast<__half*>(v_scratch_.data()), Hkv,
            one, zero);
        if (has_bias_) {
            kernels::launch_add_bias(static_cast<__half*>(v_scratch_.data()),
                            static_cast<const __half*>(b_v_.data()),
                            Mflat, Hkv, Mflat * Hkv, /*stream=*/0);
        }
    }

    // Q/K RMSNorm pre-RoPE (Gemma 3 only). Applied in-place on the
    // 4-D view that we'll feed into RoPE.
    Tensor q_4d({B, S, static_cast<int>(num_heads_), static_cast<int>(head_dim_)},
                DType::FP16, Device::cuda(device_index_));
    cudaMemcpy(q_4d.data(), q_buf_.data(),
               static_cast<int64_t>(B) * S * Hq * sizeof(__half),
               cudaMemcpyDeviceToDevice);

    Tensor k_4d({B, S, static_cast<int>(num_kv_heads_), static_cast<int>(head_dim_)},
                DType::FP16, Device::cuda(device_index_));
    cudaMemcpy(k_4d.data(), k_scratch_.data(),
               static_cast<int64_t>(B) * S * Hkv * sizeof(__half),
               cudaMemcpyDeviceToDevice);

    if (use_qk_norm_) {
        // Reshape to [B*S, head_dim] for the RMSNorm kernel (which expects
        // a 2-D input). RMSNorm can also be run in-place: we pass the
        // same buffer as both src and dst.
        const int64_t total_q = static_cast<int64_t>(B) * S * num_heads_;
        const int64_t total_k = static_cast<int64_t>(B) * S * num_kv_heads_;
        Tensor q_flat = Tensor::empty({total_q, head_dim_}, DType::FP16,
                                      Device::cuda(device_index_));
        Tensor k_flat = Tensor::empty({total_k, head_dim_}, DType::FP16,
                                      Device::cuda(device_index_));
        cudaMemcpy(q_flat.data(), q_4d.data(),
                   total_q * head_dim_ * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(k_flat.data(), k_4d.data(),
                   total_k * head_dim_ * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        kernels::launch_rmsnorm(
            static_cast<const __half*>(q_flat.data()),
            static_cast<const __half*>(w_q_norm_.data()),
            static_cast<__half*>(q_flat.data()),
            static_cast<int>(total_q), static_cast<int>(head_dim_),
            qk_norm_eps_, 0, /*stream=*/0);
        kernels::launch_rmsnorm(
            static_cast<const __half*>(k_flat.data()),
            static_cast<const __half*>(w_k_norm_.data()),
            static_cast<__half*>(k_flat.data()),
            static_cast<int>(total_k), static_cast<int>(head_dim_),
            qk_norm_eps_, 0, /*stream=*/0);
        cudaMemcpy(q_4d.data(), q_flat.data(),
                   total_q * head_dim_ * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(k_4d.data(), k_flat.data(),
                   total_k * head_dim_ * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
    }

    // RoPE on Q and K. Both use the same inv_freq but operate on different
    // tensors. We need to reshape from [B, S, H_*D] to [B, S, num_*heads, head_dim].
    // For Gemma 3 dual-band: switch to local theta before this forward if
    // this layer is a sliding layer.
    if (use_local_rope_) {
        rope_q_->set_theta_base(local_rope_theta_);
        rope_k_->set_theta_base(local_rope_theta_);
    } else {
        rope_q_->set_theta_base(rope_theta_);
        rope_k_->set_theta_base(rope_theta_);
    }
    Tensor q_roped = rope_q_->forward(q_4d, positions);
    Tensor k_roped = rope_k_->forward(k_4d, positions);

    // Copy K, V into the per-layer cache at slot [cur_len, cur_len+S).
    // kv_*_ptr layout: [num_kv_heads, max_seq, head_dim] FP16 (we treat the
    // cache as a flat buffer and index into it).
    
    // Bounds check: ensure we don't exceed max_seq
    if (cur_len + S > max_seq) {
        throw std::runtime_error("Attention::forward: sequence length " + 
                                 std::to_string(cur_len + S) + 
                                 " exceeds max_seq " + std::to_string(max_seq));
    }
    
    const __half* k_src = static_cast<const __half*>(k_roped.data());
    const __half* v_src = static_cast<const __half*>(v_scratch_.data());
    const int per_step_bytes = static_cast<int>(num_kv_heads_ * head_dim_) * 2;
    for (int s = 0; s < S; ++s) {
        MI_CHECK_CUDA(cudaMemcpyAsync(
            kv_k_ptr + (cur_len + s) * num_kv_heads_ * head_dim_,
            k_src + s * num_kv_heads_ * head_dim_,
            per_step_bytes, cudaMemcpyDeviceToDevice, /*stream=*/0));
        MI_CHECK_CUDA(cudaMemcpyAsync(
            kv_v_ptr + (cur_len + s) * num_kv_heads_ * head_dim_,
            v_src + s * num_kv_heads_ * head_dim_,
            per_step_bytes, cudaMemcpyDeviceToDevice, /*stream=*/0));
    }
    MI_CHECK_CUDA(cudaDeviceSynchronize());

    // Naive attention: K, V now contain history [0, cur_len+S).
    const int S_k = static_cast<int>(cur_len + S);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const int num_kv_groups = static_cast<int>(num_heads_ / num_kv_heads_);
    kernels::launch_naive_attn(
        static_cast<const __half*>(q_roped.data()),
        kv_k_ptr,
        kv_v_ptr,
        static_cast<__half*>(attn_out_buf_.data()),
        B, S, S_k,
        static_cast<int>(num_heads_),
        static_cast<int>(num_kv_heads_),
        static_cast<int>(head_dim_),
        num_kv_groups, scale, is_prefill ? 1 : 0,
        static_cast<int>(sliding_window_),
        /*stream=*/0);

    // O projection: out = attn_out @ W_o^T
    Tensor out = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));
    {
        const int Mflat = B * S;
        gemm_rowmajor(handle, Mflat, H, Hq,
            static_cast<const __half*>(attn_out_buf_.data()), Hq,
            static_cast<const __half*>(w_o_.data()), Hq,
            static_cast<__half*>(out.data()), H,
            one, zero);
    }
    return out;
}

namespace {
inline void cuda_check_pg(cudaError_t e, const char* expr, const char* f, int l) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error ") + expr + " at " +
                                 f + ":" + std::to_string(l) + " : " +
                                 cudaGetErrorString(e));
    }
}
#define MI_CHECK_CUDA_PG(expr) cuda_check_pg((expr), #expr, __FILE__, __LINE__)
}  // namespace

Tensor Attention::forward_paged(const Tensor& hidden_states,
                                const std::vector<int64_t>& positions,
                                PagedKVCache& paged_kv,
                                int seq_id,
                                int layer_idx,
                                bool is_prefill) {
    if (hidden_states.dtype() != DType::FP16) {
        throw std::runtime_error("Attention::forward_paged expects FP16 input");
    }
    if (hidden_states.ndim() != 3) {
        throw std::runtime_error("Attention::forward_paged expects 3-D [B,S,H]");
    }
    // Single-sequence path: positions.size() == B*S where B == 1.
    // For batched decode, callers should use forward_paged_batched.
    if (hidden_states.shape()[0] != 1) {
        throw std::runtime_error(
            "Attention::forward_paged: B must be 1; use forward_paged_batched for B>1");
    }
    const int B = 1;
    const int S = static_cast<int>(hidden_states.shape()[1]);
    const int H = static_cast<int>(hidden_);
    const int Hq  = static_cast<int>(num_heads_    * head_dim_);
    const int Hkv = static_cast<int>(num_kv_heads_ * head_dim_);
    const int num_kv_groups = static_cast<int>(num_heads_ / num_kv_heads_);

    // Lazily (re)allocate persistent buffers.
    auto ensure = [&](Tensor& buf, const std::vector<int64_t>& sh, DType dt) {
        if (buf.numel() != sh[0] * sh[1] * sh[2]) {
            buf = Tensor::empty(sh, dt, Device::cuda(device_index_));
        }
    };
    ensure(q_buf_,         {B, S, Hq},  DType::FP16);
    ensure(k_scratch_,     {B, S, Hkv}, DType::FP16);
    ensure(v_scratch_,     {B, S, Hkv}, DType::FP16);
    ensure(attn_out_buf_,  {B, S, Hq},  DType::FP16);

    auto handle = get_cublas(cublas_handle_);
    const float one  = 1.0f;
    const float zero = 0.0f;

    // ---- QKV projections (same as naive) -------------------------------
    {
        const int Mflat = B * S;
        gemm_rowmajor(handle, Mflat, Hq, H,
            static_cast<const __half*>(hidden_states.data()), H,
            static_cast<const __half*>(w_q_.data()), H,
            static_cast<__half*>(q_buf_.data()), Hq,
            one, zero);
        if (has_bias_) {
            kernels::launch_add_bias(static_cast<__half*>(q_buf_.data()),
                static_cast<const __half*>(b_q_.data()),
                Mflat, Hq, Mflat * Hq, /*stream=*/0);
        }
    }
    {
        const int Mflat = B * S;
        gemm_rowmajor(handle, Mflat, Hkv, H,
            static_cast<const __half*>(hidden_states.data()), H,
            static_cast<const __half*>(w_k_.data()), H,
            static_cast<__half*>(k_scratch_.data()), Hkv,
            one, zero);
        if (has_bias_) {
            kernels::launch_add_bias(static_cast<__half*>(k_scratch_.data()),
                static_cast<const __half*>(b_k_.data()),
                Mflat, Hkv, Mflat * Hkv, /*stream=*/0);
        }
        gemm_rowmajor(handle, Mflat, Hkv, H,
            static_cast<const __half*>(hidden_states.data()), H,
            static_cast<const __half*>(w_v_.data()), H,
            static_cast<__half*>(v_scratch_.data()), Hkv,
            one, zero);
        if (has_bias_) {
            kernels::launch_add_bias(static_cast<__half*>(v_scratch_.data()),
                static_cast<const __half*>(b_v_.data()),
                Mflat, Hkv, Mflat * Hkv, /*stream=*/0);
        }
    }

    // ---- Q/K RMSNorm (Gemma 3) and RoPE ---------------------------------
    q_4d_paged_ = Tensor({B, S, static_cast<int>(num_heads_), static_cast<int>(head_dim_)},
                          DType::FP16, Device::cuda(device_index_));
    MI_CHECK_CUDA_PG(cudaMemcpy(q_4d_paged_.data(), q_buf_.data(),
        static_cast<int64_t>(B) * S * Hq * sizeof(__half),
        cudaMemcpyDeviceToDevice));
    k_4d_paged_ = Tensor({B, S, static_cast<int>(num_kv_heads_), static_cast<int>(head_dim_)},
                          DType::FP16, Device::cuda(device_index_));
    MI_CHECK_CUDA_PG(cudaMemcpy(k_4d_paged_.data(), k_scratch_.data(),
        static_cast<int64_t>(B) * S * Hkv * sizeof(__half),
        cudaMemcpyDeviceToDevice));

    if (use_qk_norm_) {
        // Apply Q/K RMSNorm in-place on the 4-D view (head_dim is the last
        // dimension, contiguous). launch_rmsnorm takes [N, D] 2-D; we pass
        // q_4d_paged_ data as both src and dst (in-place).
        const int64_t total_q = static_cast<int64_t>(B) * S * num_heads_;
        const int64_t total_k = static_cast<int64_t>(B) * S * num_kv_heads_;
        kernels::launch_rmsnorm(
            static_cast<const __half*>(q_4d_paged_.data()),
            static_cast<const __half*>(w_q_norm_.data()),
            static_cast<__half*>(q_4d_paged_.data()),
            static_cast<int>(total_q), static_cast<int>(head_dim_),
            qk_norm_eps_, 0, /*stream=*/0);
        kernels::launch_rmsnorm(
            static_cast<const __half*>(k_4d_paged_.data()),
            static_cast<const __half*>(w_k_norm_.data()),
            static_cast<__half*>(k_4d_paged_.data()),
            static_cast<int>(total_k), static_cast<int>(head_dim_),
            qk_norm_eps_, 0, /*stream=*/0);
    }

    // Gemma 3 dual-band RoPE switch.
    if (use_local_rope_) {
        rope_q_->set_theta_base(local_rope_theta_);
        rope_k_->set_theta_base(local_rope_theta_);
    } else {
        rope_q_->set_theta_base(rope_theta_);
        rope_k_->set_theta_base(rope_theta_);
    }
    Tensor q_roped = rope_q_->forward(q_4d_paged_, positions);
    Tensor k_roped = rope_k_->forward(k_4d_paged_, positions);

    // ---- Scatter K, V into the paged cache (one token at a time) ------
    // The kernel expects [B, S, H_kv, D] layout; we have [B, S, H_kv*D]
    // contiguous which is exactly that layout in memory.
    //
    // Reuse the (lazily allocated) per-(max_blocks_per_seq, num_kv_heads,
    // head_dim) block_table scratch on device.
    const BlockTable& tbl = paged_kv.table(seq_id);
    const int num_blocks_used = static_cast<int>(tbl.block_ids.size());
    const int seq_len = paged_kv.num_tokens(seq_id);
    const int start_pos = seq_len - S;   // global position of the first new token
    const int max_blocks_per_seq = paged_kv.max_blocks_per_seq();

    if (cached_max_blocks_ != max_blocks_per_seq
        || cached_num_kv_heads_ != num_kv_heads_
        || cached_head_dim_     != head_dim_) {
        block_table_dev_     = Tensor::empty({max_blocks_per_seq}, DType::INT32,
                                             Device::cuda(device_index_));
        num_blocks_used_dev_ = Tensor::empty({1}, DType::INT32,
                                             Device::cuda(device_index_));
        seq_len_dev_         = Tensor::empty({1}, DType::INT32,
                                             Device::cuda(device_index_));
        cached_max_blocks_    = max_blocks_per_seq;
        cached_num_kv_heads_ = static_cast<int>(num_kv_heads_);
        cached_head_dim_     = static_cast<int>(head_dim_);
    }
    // Host->device copies of metadata.
    MI_CHECK_CUDA_PG(cudaMemcpy(block_table_dev_.data(),
        tbl.block_ids.data(), max_blocks_per_seq * sizeof(int),
        cudaMemcpyHostToDevice));
    MI_CHECK_CUDA_PG(cudaMemcpy(num_blocks_used_dev_.data(),
        &num_blocks_used, sizeof(int), cudaMemcpyHostToDevice));
    MI_CHECK_CUDA_PG(cudaMemcpy(seq_len_dev_.data(),
        &seq_len, sizeof(int), cudaMemcpyHostToDevice));
    // start_pos is a single int for the single-sequence path. Use a 1-elt
    // scratch buffer (allocated lazily alongside the others).
    if (!start_pos_dev_.numel()) {
        start_pos_dev_ = Tensor::empty({1}, DType::INT32,
                                       Device::cuda(device_index_));
    }
    MI_CHECK_CUDA_PG(cudaMemcpy(start_pos_dev_.data(), &start_pos,
        sizeof(int), cudaMemcpyHostToDevice));

    auto* k_cache_data = static_cast<__half*>(paged_kv.allocator().k_block_ptr(
        0, 0));    // layer offset is folded into the kernel's blk_base calc
    auto* v_cache_data = static_cast<__half*>(paged_kv.allocator().v_block_ptr(
        0, 0));
    const int num_blocks_pool = paged_kv.allocator().num_blocks();

    kernels::launch_paged_kv_scatter(
        static_cast<const __half*>(k_roped.data()),
        k_cache_data,
        static_cast<const int*>(block_table_dev_.data()),
        static_cast<const int*>(seq_len_dev_.data()),
        static_cast<const int*>(start_pos_dev_.data()),
        B, S, layer_idx, num_blocks_pool,
        static_cast<int>(num_kv_heads_),
        static_cast<int>(head_dim_),
        BlockAllocator::kBlockSize,
        max_blocks_per_seq,
        /*stream=*/0);
    kernels::launch_paged_kv_scatter(
        static_cast<const __half*>(v_scratch_.data()),
        v_cache_data,
        static_cast<const int*>(block_table_dev_.data()),
        static_cast<const int*>(seq_len_dev_.data()),
        static_cast<const int*>(start_pos_dev_.data()),
        B, S, layer_idx, num_blocks_pool,
        static_cast<int>(num_kv_heads_),
        static_cast<int>(head_dim_),
        BlockAllocator::kBlockSize,
        max_blocks_per_seq,
        /*stream=*/0);

    // ---- PagedAttention ------------------------------------------------
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    kernels::launch_paged_attn(
        static_cast<const __half*>(q_roped.data()),
        k_cache_data, v_cache_data,
        static_cast<const int*>(block_table_dev_.data()),
        static_cast<const int*>(num_blocks_used_dev_.data()),
        static_cast<const int*>(seq_len_dev_.data()),
        static_cast<const int*>(start_pos_dev_.data()),
        B, S,
        static_cast<int>(num_heads_),
        static_cast<int>(num_kv_heads_),
        static_cast<int>(head_dim_),
        num_kv_groups,
        max_blocks_per_seq,
        layer_idx,
        num_blocks_pool,
        scale,
        is_prefill ? 1 : 0,
        static_cast<int>(sliding_window_),
        static_cast<__half*>(attn_out_buf_.data()),
        /*stream=*/0);

    // ---- O projection --------------------------------------------------
    Tensor out = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));
    {
        const int Mflat = B * S;
        gemm_rowmajor(handle, Mflat, H, Hq,
            static_cast<const __half*>(attn_out_buf_.data()), Hq,
            static_cast<const __half*>(w_o_.data()), Hq,
            static_cast<__half*>(out.data()), H,
            one, zero);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Batched paged forward (Week 5+, decode batch).
//
// Same shape as forward_paged but accepts B >= 1 with independent sequences.
// ---------------------------------------------------------------------------
Tensor Attention::forward_paged_batched(const Tensor& hidden_states,
                                        const std::vector<int64_t>& positions,
                                        PagedKVCache& paged_kv,
                                        const std::vector<int>& seq_ids,
                                        const std::vector<int>& start_pos,
                                        int layer_idx,
                                        bool is_prefill) {
    if (hidden_states.dtype() != DType::FP16) {
        throw std::runtime_error("forward_paged_batched: FP16 input required");
    }
    if (hidden_states.ndim() != 3) {
        throw std::runtime_error("forward_paged_batched: 3-D [B,S,H] required");
    }
    const int B = static_cast<int>(hidden_states.shape()[0]);
    const int S = static_cast<int>(hidden_states.shape()[1]);
    if (B <= 0) return Tensor::empty({0, 0, static_cast<int>(hidden_)},
                                      DType::FP16, Device::cuda(device_index_));
    if (static_cast<int>(seq_ids.size()) != B) {
        throw std::runtime_error("forward_paged_batched: seq_ids size mismatch");
    }
    if (static_cast<int>(start_pos.size()) != B) {
        throw std::runtime_error("forward_paged_batched: start_pos size mismatch");
    }
    if (static_cast<int>(positions.size()) != B * S) {
        throw std::runtime_error("forward_paged_batched: positions size mismatch");
    }

    const int H   = static_cast<int>(hidden_);
    const int Hq  = static_cast<int>(num_heads_    * head_dim_);
    const int Hkv = static_cast<int>(num_kv_heads_ * head_dim_);
    const int num_kv_groups = static_cast<int>(num_heads_ / num_kv_heads_);

    auto ensure = [&](Tensor& buf, const std::vector<int64_t>& sh, DType dt) {
        if (buf.numel() != sh[0] * sh[1] * sh[2]) {
            buf = Tensor::empty(sh, dt, Device::cuda(device_index_));
        }
    };
    ensure(q_buf_,           {B, S, Hq},  DType::FP16);
    ensure(k_scratch_,       {B, S, Hkv}, DType::FP16);
    ensure(v_scratch_,       {B, S, Hkv}, DType::FP16);
    ensure(attn_out_b_,      {B, S, Hq},  DType::FP16);

    auto handle = get_cublas(cublas_handle_);
    const float one  = 1.0f;
    const float zero = 0.0f;

    // ---- QKV projections -----------------------------------------------
    const int Mflat = B * S;
    gemm_rowmajor(handle, Mflat, Hq, H,
        static_cast<const __half*>(hidden_states.data()), H,
        static_cast<const __half*>(w_q_.data()), H,
        static_cast<__half*>(q_buf_.data()), Hq, one, zero);
    if (has_bias_) {
        kernels::launch_add_bias(static_cast<__half*>(q_buf_.data()),
            static_cast<const __half*>(b_q_.data()),
            Mflat, Hq, Mflat * Hq, /*stream=*/0);
    }
    gemm_rowmajor(handle, Mflat, Hkv, H,
        static_cast<const __half*>(hidden_states.data()), H,
        static_cast<const __half*>(w_k_.data()), H,
        static_cast<__half*>(k_scratch_.data()), Hkv, one, zero);
    if (has_bias_) {
        kernels::launch_add_bias(static_cast<__half*>(k_scratch_.data()),
            static_cast<const __half*>(b_k_.data()),
            Mflat, Hkv, Mflat * Hkv, /*stream=*/0);
    }
    gemm_rowmajor(handle, Mflat, Hkv, H,
        static_cast<const __half*>(hidden_states.data()), H,
        static_cast<const __half*>(w_v_.data()), H,
        static_cast<__half*>(v_scratch_.data()), Hkv, one, zero);
    if (has_bias_) {
        kernels::launch_add_bias(static_cast<__half*>(v_scratch_.data()),
            static_cast<const __half*>(b_v_.data()),
            Mflat, Hkv, Mflat * Hkv, /*stream=*/0);
    }

    // ---- Q/K RMSNorm (Gemma 3) and RoPE (positions is B*S long) -------
    q_4d_b_ = Tensor({B, S, static_cast<int>(num_heads_), static_cast<int>(head_dim_)},
                     DType::FP16, Device::cuda(device_index_));
    MI_CHECK_CUDA_PG(cudaMemcpy(q_4d_b_.data(), q_buf_.data(),
        static_cast<int64_t>(B) * S * Hq * sizeof(__half),
        cudaMemcpyDeviceToDevice));
    k_4d_b_ = Tensor({B, S, static_cast<int>(num_kv_heads_), static_cast<int>(head_dim_)},
                     DType::FP16, Device::cuda(device_index_));
    MI_CHECK_CUDA_PG(cudaMemcpy(k_4d_b_.data(), k_scratch_.data(),
        static_cast<int64_t>(B) * S * Hkv * sizeof(__half),
        cudaMemcpyDeviceToDevice));

    if (use_qk_norm_) {
        const int64_t total_q = static_cast<int64_t>(B) * S * num_heads_;
        const int64_t total_k = static_cast<int64_t>(B) * S * num_kv_heads_;
        kernels::launch_rmsnorm(
            static_cast<const __half*>(q_4d_b_.data()),
            static_cast<const __half*>(w_q_norm_.data()),
            static_cast<__half*>(q_4d_b_.data()),
            static_cast<int>(total_q), static_cast<int>(head_dim_),
            qk_norm_eps_, 0, /*stream=*/0);
        kernels::launch_rmsnorm(
            static_cast<const __half*>(k_4d_b_.data()),
            static_cast<const __half*>(w_k_norm_.data()),
            static_cast<__half*>(k_4d_b_.data()),
            static_cast<int>(total_k), static_cast<int>(head_dim_),
            qk_norm_eps_, 0, /*stream=*/0);
    }

    // Gemma 3 dual-band RoPE switch.
    if (use_local_rope_) {
        rope_q_->set_theta_base(local_rope_theta_);
        rope_k_->set_theta_base(local_rope_theta_);
    } else {
        rope_q_->set_theta_base(rope_theta_);
        rope_k_->set_theta_base(rope_theta_);
    }
    Tensor q_roped = rope_q_->forward_batched(q_4d_b_, positions);
    Tensor k_roped = rope_k_->forward_batched(k_4d_b_, positions);

    // ---- Build per-sequence metadata arrays on device ----------------
    const int max_blocks_per_seq = paged_kv.max_blocks_per_seq();
    std::vector<int> bt_flat(B * max_blocks_per_seq, 0);
    std::vector<int> nbu_h(B), sl_h(B);
    for (int b = 0; b < B; ++b) {
        const BlockTable& t = paged_kv.table(seq_ids[b]);
        for (size_t i = 0; i < t.block_ids.size(); ++i) {
            bt_flat[b * max_blocks_per_seq + i] = t.block_ids[i];
        }
        nbu_h[b] = static_cast<int>(t.block_ids.size());
        sl_h[b]  = paged_kv.num_tokens(seq_ids[b]);
    }
    if (cached_max_B_ < B) {
        block_tables_dev_      = Tensor::empty({B * max_blocks_per_seq},
                                                DType::INT32,
                                                Device::cuda(device_index_));
        num_blocks_used_b_dev_ = Tensor::empty({B}, DType::INT32,
                                                Device::cuda(device_index_));
        seq_len_b_dev_         = Tensor::empty({B}, DType::INT32,
                                                Device::cuda(device_index_));
        start_pos_b_dev_       = Tensor::empty({B}, DType::INT32,
                                                Device::cuda(device_index_));
        cached_max_B_ = B;
    }
    MI_CHECK_CUDA_PG(cudaMemcpy(block_tables_dev_.data(),
        bt_flat.data(), B * max_blocks_per_seq * sizeof(int),
        cudaMemcpyHostToDevice));
    MI_CHECK_CUDA_PG(cudaMemcpy(num_blocks_used_b_dev_.data(),
        nbu_h.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    MI_CHECK_CUDA_PG(cudaMemcpy(seq_len_b_dev_.data(),
        sl_h.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    MI_CHECK_CUDA_PG(cudaMemcpy(start_pos_b_dev_.data(),
        start_pos.data(), B * sizeof(int), cudaMemcpyHostToDevice));

    auto* k_cache_data = static_cast<__half*>(paged_kv.allocator().k_block_ptr(
        0, 0));
    auto* v_cache_data = static_cast<__half*>(paged_kv.allocator().v_block_ptr(
        0, 0));
    const int num_blocks_pool = paged_kv.allocator().num_blocks();

    // ---- Scatter K, V into the paged cache ----------------------------
    kernels::launch_paged_kv_scatter(
        static_cast<const __half*>(k_roped.data()),
        k_cache_data,
        static_cast<const int*>(block_tables_dev_.data()),
        static_cast<const int*>(seq_len_b_dev_.data()),
        static_cast<const int*>(start_pos_b_dev_.data()),
        B, S, layer_idx, num_blocks_pool,
        static_cast<int>(num_kv_heads_),
        static_cast<int>(head_dim_),
        BlockAllocator::kBlockSize,
        max_blocks_per_seq,
        /*stream=*/0);
    kernels::launch_paged_kv_scatter(
        static_cast<const __half*>(v_scratch_.data()),
        v_cache_data,
        static_cast<const int*>(block_tables_dev_.data()),
        static_cast<const int*>(seq_len_b_dev_.data()),
        static_cast<const int*>(start_pos_b_dev_.data()),
        B, S, layer_idx, num_blocks_pool,
        static_cast<int>(num_kv_heads_),
        static_cast<int>(head_dim_),
        BlockAllocator::kBlockSize,
        max_blocks_per_seq,
        /*stream=*/0);

    // ---- PagedAttention (per-sequence, batched launch) ---------------
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    kernels::launch_paged_attn(
        static_cast<const __half*>(q_roped.data()),
        k_cache_data, v_cache_data,
        static_cast<const int*>(block_tables_dev_.data()),
        static_cast<const int*>(num_blocks_used_b_dev_.data()),
        static_cast<const int*>(seq_len_b_dev_.data()),
        static_cast<const int*>(start_pos_b_dev_.data()),
        B, S,
        static_cast<int>(num_heads_),
        static_cast<int>(num_kv_heads_),
        static_cast<int>(head_dim_),
        num_kv_groups,
        max_blocks_per_seq,
        layer_idx,
        num_blocks_pool,
        scale,
        is_prefill ? 1 : 0,
        static_cast<int>(sliding_window_),
        static_cast<__half*>(attn_out_b_.data()),
        /*stream=*/0);

    // ---- O projection -------------------------------------------------
    Tensor out = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));
    gemm_rowmajor(handle, Mflat, H, Hq,
        static_cast<const __half*>(attn_out_b_.data()), Hq,
        static_cast<const __half*>(w_o_.data()), Hq,
        static_cast<__half*>(out.data()), H, one, zero);
    return out;
}

}  // namespace mini_infer