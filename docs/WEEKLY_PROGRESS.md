# mini-infer 周进度记录

> 每周交付物 + 验收记录 + 踩坑记录

---

## Week 1 — 基础设施（2026-07-20 ~ 07-26）

**目标**：搭建 CUDA 开发环境，建立项目骨架，跑通最小 CUDA kernel。

### 交付清单

- [x] Git 仓库初始化（git init + .gitignore + LICENSE + README）
- [x] `CMakeLists.txt`（CUDAToolkit + C++17，编译 hello kernel 通过，sm_86）
- [x] `examples/CMakeLists.txt` + `tests/CMakeLists.txt` aggregator
- [x] `src/core/tensor.h/cpp` — Tensor 抽象（FP16/FP32，h2d/d2h/d2d，stride，contiguous）
- [x] `src/core/allocator.h/cpp` — BumpAllocator（256B 对齐，reset，peak 统计）
- [x] `examples/00_hello_cuda` — 打印 GPU 信息 + 跑通 kernel
- [x] `tests/test_tensor` — 5 个测试：metadata / CPU fill / CUDA zero / h2d-d2h / add kernel
- [x] `tests/test_allocator` — 5 个测试：256B 对齐 / 单调不重叠 / GPU 写入 / reset / 溢出

### 验收命令

```bash
export PATH=/data1/tyh/miniconda3/bin:/usr/local/cuda-12.1/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j
./build/examples/00_hello_cuda/00_hello_cuda     # 输出 GPU 信息
cd build && ctest --output-on-failure            # 2/2 passed
```

### 验收结果

```
mini-infer :: hello CUDA  (Week 1)
Detected 4 CUDA device(s)
Using device 0 : NVIDIA RTX A6000
  Compute capability : 8.6
  SM count           : 84
  Total global mem   : 47.54 GiB
  Driver version     : 12.2
  Runtime version    : 12.1
  Warp size          : 32
Kernel roundtrip: 1024 / 1024 correct

ctest> 100% tests passed, 0 tests failed out of 2
```

### 技术决策

- **CUDA 版本**：使用本地 `/usr/local/cuda-12.1/`（驱动 12.2 向后兼容 12.1 toolkit）。
- **CMake Generator**：Ninja（在 tyh conda env 中）。`find_package(CUDAToolkit)` + `CUDA::cudart`。
- **GPU arch**：A6000 = sm_86。`CMAKE_CUDA_ARCHITECTURES` 在 `find_package` 之前用 `CACHE FORCE` 设置，避免被默认 52 覆盖。
- **CUDA 文件扩展名**：kernel launch 语法 (`<<<>>>`) 与 nvcc intrinsic (`blockIdx` 等) 必须在 nvcc 下编译。采用 `.cpp` 后缀但通过 `set_source_files_properties(... LANGUAGE CUDA)` 强制走 nvcc。
- **Tensor dtype / device**：`DType` (FP32/FP16/BF16/INT32/INT64) + `Device` (CPU/CUDA) 枚举；`void* data_` + `nbytes_` 存储，避免模板膨胀。
- **Tensor stride**：`std::vector<int64_t>`，默认 row-major contiguous；`is_contiguous()` 检查；后续 PagedAttention 会用 stride 表达非连续视图。
- **Allocator**：256-byte 对齐（cuBLAS 要求），`peak` 统计 + `reset()`。Week 5 会扩展为 KV Cache block pool。
- **FP16 host 转换**：手写 IEEE-754 f32→f16 位转换（避免在 `.cpp` 中引入 `cuda_fp16.h`，后者需 nvcc 编译）。
- **CUDA error 处理**：`MI_CHECK_CUDA(expr)` 宏抛 `std::runtime_error`，含行号 + 表达式文本。

### 踩坑

1. **CMake 默认 CUDA arch 被覆盖**：直接 `set(CMAKE_CUDA_ARCHITECTURES 86)` 放在 `find_package` 之后会被 Toolkit 模块设为 52。修复：放在 `find_package` 之前 + `CACHE ... FORCE`。
2. **C++ 文件中的 CUDA 语法**：`blockIdx` / `<<<>>>` 不能被 GCC 解析。修复：`set_source_files_properties(... LANGUAGE CUDA)`。
3. **GCC 9 字符串字面量拼接**：`"text " + ptr` 不合法，需先 `std::string("text ") + ptr`。
4. **Device struct 没有 `device_` 字段**：字段是 `type` 和 `index`；`Tensor::to()` 中应为 `target`，不是 `target.device_`。

### 文件清单

```
mini-infer/
├── CMakeLists.txt                       # root
├── .gitignore  LICENSE  README.md
├── docs/
│   ├── PROJECT_PLAN.md                  # 8 周路线图 + 消融实验设计
│   └── WEEKLY_PROGRESS.md               # 本文件
├── src/core/
│   ├── tensor.h     tensor.cpp          # Tensor
│   ├── allocator.h  allocator.cpp       # BumpAllocator
├── examples/
│   ├── CMakeLists.txt                   # aggregator
│   └── 00_hello_cuda/
│       ├── CMakeLists.txt
│       └── main.cpp                     # GPU info + hello kernel
└── tests/
    ├── CMakeLists.txt                   # aggregator
    ├── test_tensor/{CMakeLists.txt, main.cpp}   # 5 tests
    └── test_allocator/{CMakeLists.txt, main.cpp}# 5 tests
```

---

## Week 2 — 基础算子（2026-07-27 ~ 08-02）

**目标**：实现所有基础 CUDA kernel，每个 kernel 都要和 PyTorch 对拍验证。

### 交付清单

- [x] `kernels/rmsnorm_kernel.{cuh,cu}` — RMSNorm，warp shuffle 归约，FP16 存储 FP32 累加
- [x] `kernels/rope_kernel.{cuh,cu}` — RoPE + cos/sin 预计算
- [x] `kernels/softmax_kernel.{cuh,cu}` — 两遍 softmax（max + sum），数值稳定
- [x] `kernels/swiglu_kernel.{cuh,cu}` — fused silu(gate) * up
- [x] `layers/{rmsnorm,rope,mlp}.{h,cpp}` — 高层封装 + FP16/CUDA 校验
- [x] `kernels/warp_reduce.cuh` — warp / block 级 sum/max 归约
- [x] GEMM：`layers/mlp.cpp` 中调 `cublasGemmEx`，FP16 输入 FP32 累加
- [x] `tests/verify/verify.py` — 通用 torch 对拍脚本（6 个 op）
- [x] `tests/test_{rmsnorm,rope,softmax,swiglu,mlp}/` — 5 个 kernel 单测，全部 < 1e-3 误差
- [x] `benchmarks/bench_kernels.cpp` — 单 kernel cudaEvent 计时
- [x] `scripts/{build,run_tests}.sh` — 一键构建/测试（自动设环境变量）

### 验收结果

```
$ scripts/run_tests.sh
100% tests passed, 0 tests failed out of 7

$ build/benchmarks/bench_kernels
  rmsnorm N=4 D=3584              median=   6.14 us   p99=   8.19 us
  rope B=4 S=512 H=28 D=128       median=  67.58 us   p99=  83.97 us
  softmax N=57344 D=512           median= 176.13 us   p99= 191.49 us
  swiglu N=2048 I=18944           median= 348.16 us   p99= 349.18 us
```

### 关键技术点

- **RMSNorm**：warp shuffle 归约 + 一个 block-reduce 经 shared memory 收尾；
  `rsqrtf(sum/D + eps)` 一次性拿到倒数避免一次除法。
- **RoPE**：cos/sin 表形状 `[S, head_dim/2]`，kernel 按 `i < half` 遍历，同时更新
  `y[i]` 与 `y[i+half]`。Qwen2.5 theta_base = 1000000。
- **Softmax**：两遍 form，先算 max 再算 sum；分母 `1/sum` 一次算后广播。
  Week 5 才会用 online 单遍形式（FlashAttention 用）。
- **cuBLAS row-major trick**：`row_major(X)[i,j] == col_major(X^T)[j,i]`，
  cuBLAS 看到的列主矩阵是我们行主矩阵的转置。要让 cuBLAS 算出
  `C = A @ B`，需要 `OP_T` on cuBLAS-A (= our-B)，`OP_N` on cuBLAS-B (= our-A)。
- **fp32 累加**：所有 kernel / GEMM 用 `CUBLAS_COMPUTE_32F`，FP16 仅做存储。

### 踩坑

1. **GEMM 错误的 layout**：第一版用 `OP_N` 双边，输出量级偏离 torch 参考 80×。
   原因：cuBLAS 看到的 "row-B(N,K)" 是列主 (K, N)，需要 `OP_T` 才能拿到真正的
   `B` 矩阵。修复后 max_abs < 5e-4 vs torch。
2. **`#include "foo_kernel.cu"` in `foo.cpp`**：.cu 里的 `blockIdx` 等让 g++ 编译失败。
   重构为 `.cuh`（声明） + `.cu`（实现）分离，layers/*.cpp 只 include `.cuh`。
3. **`cast_to_f16` 模板显式特化在 nvcc 下报错**：nvcc 不允许带 `static` 的显式
   特化。改成非模板静态函数。
4. **链接时找不到 libm**：conda gcc 10.4 的 sysroot 缺 `libm.so.6`，
   链接器走 `/usr/bin/ld` 但搜索路径是 conda 的 sysroot。
   解决：显式 `target_link_libraries(... m)`，并设置 `CC/CXX/CUDAHOSTCXX` 指向
   系统 gcc 9.3。
5. **ctest 工作目录**：cmake 默认在 `build/tests/<test>/`，python verifier
   用相对路径 `tests/verify/verify.py` 找不到。加 `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}`。
6. **rope Python 广播**：`x[B,S,H,half] * cos[S,half]` 默认右对齐得到
   `[1,1,S,half]`，与 `[B,S,H,half]` 在 H 维冲突。改成 `cos.reshape(S,half).reshape(1,S,1,half)`。

### 文件清单（本周末）

```
src/
├── kernels/
│   ├── warp_reduce.cuh           # warp/block 归约原语
│   ├── rmsnorm_kernel.{cuh,cu}
│   ├── rope_kernel.{cuh,cu}      # 含 cos/sin 预计算 kernel
│   ├── softmax_kernel.{cuh,cu}
│   └── swiglu_kernel.{cuh,cu}
└── layers/
    ├── rmsnorm.{h,cpp}
    ├── rope.{h,cpp}
    └── mlp.{h,cpp}               # 含 cuBLAS GEMM 行主适配

tests/
├── verify/verify.py              # 通用 torch 对拍脚本
├── test_rmsnorm/
├── test_rope/
├── test_softmax/
├── test_swiglu/
└── test_mlp/

benchmarks/bench_kernels.cpp      # cudaEvent 单 kernel 计时
scripts/{build,run_tests}.sh      # 一键 build + ctest
```
---

## Week 3 — 模型加载 + 计算图（2026-08-03 ~ 08-09）

**目标**：能加载 Qwen2.5-7B-Instruct 的权重，构建计算图，但还没跑通推理。

### 交付清单

- [x] `src/model/model_config.{h,cpp}` — 从 `config.json` 解析 num_layers/hidden_size/num_attention_heads/num_key_value_heads 等
- [x] `src/model/safetensors_loader.{h,cpp}` — 多 shard 加载，mmap，懒打开
- [x] `src/model/qwen_model.{h,cpp}` — 28 层 + MLP + RMSNorm，BF16→FP16 上传到 GPU
- [x] `src/core/graph.{h,cpp}` — 线性链计算图（add_input / add_node / add_output / topo_order / print）
- [x] `scripts/inspect_weights.py` — torch 实现的对照工具
- [x] `tests/test_model_config` — 4 个用例（valid Qwen2.5 / 缺字段默认值 / GQA 不整除 / 缺必填字段）
- [x] `tests/test_graph` — 空图 / 单节点 / Qwen 单 block / 打印
- [x] `tests/test_safetensors` — 339 个张量索引 / 12 个关键张量 shape 校验 / BF16 round-trip / missing→nullptr
- [x] `tests/test_qwen_model` — 加载完整 14GB Qwen2.5-Coder-7B-Instruct，FP16 上传 GPU，summarize 与 inspect_weights.py 一致
- [x] `third_party/nlohmann/` — vendored single-header JSON

### 验收结果

```
$ scripts/run_tests.sh
100% tests passed, 0 tests failed out of 11
```

```
$ scripts/inspect_weights.py /data1/kdy/LLMs/Qwen2.5-Coder-7B-Instruct \
    --filter "layers.0."
model.layers.0.input_layernorm.weight    shape=[3584]  dtype=bfloat16  mean=+2.4788e-01  std=2.0465e-02
model.layers.0.post_attention_layernorm.weight  shape=[3584]  mean=+1.9127e-01  std=5.3242e-02
model.layers.0.self_attn.q_proj.weight   shape=[3584,3584]  mean=-2.3098e-04  std=1.7593e-02
... (12 行)
```

C++ 端 summarize 输出（同一层）：
```
model.layers.0.input_layernorm.weight    shape=[3584]   mean=0.247882  std=0.0204618
model.layers.0.post_attention_layernorm.weight  shape=[3584]  mean=0.191265  std=0.0532344
model.layers.0.self_attn.q_proj.weight   shape=[3584,3584]  mean=-1.90e-05  std=0.0158129
```

均值/方差在 BF16→FP16 的 round-trip 精度内完全一致。

### 关键技术点

- **safetensors 格式**：前 8 字节 header 长度（小端 uint64） → JSON → tensor 数据；
  JSON 内 `data_offsets` 是相对**数据段**（不是文件）的绝对字节偏移。
- **mmap 而非 read**：14GB 模型 mmap 后只占 page cache，不膨胀 RSS。WeightIndex
  对 shard 懒打开，第一次访问某个 tensor 才打开对应 shard。
- **BF16→FP16 CPU 转换**：BF16 是 FP32 的高 16 位。CPU 循环逐元素左移 16 位
  得到 FP32 bits，再走已有的 f32→f16 round-to-nearest-even 转换器。
- **GQA**：Qwen2.5-7B 是 `num_attention_heads=28, num_key_value_heads=4`，
  KV cache 按 kv heads 算。`ModelConfig::head_dim()` / `kv_head_dim()` / `num_kv_groups()`。
- **Qwen2 的 QKV bias**：Qwen2 / Qwen2.5 架构每个 q/k/v projection 都有 bias；
  LLaMA 没有，loader 用 `idx.find("...q_proj.bias")` 检查。
- **tie_word_embeddings**：7B 是 false（独立 lm_head），0.5B 可能是 true；
  当前实现：true 时 alias lm_head_ = embed_，false 时单独加载。

### 踩坑

1. **Tensor 深拷贝 vs 浅拷贝**：Week 1 设计成 move-only，但 Week 3 model loader
   需要频繁在函数间按值返回（helper functions）。改成深拷贝（allocate + cudaMemcpy），
   但 cuBLAS handle 不允许共享 → MLP / RMSNorm 改成 move-only，禁用 copy。
2. **MLP 的 cuBLAS handle 双重释放**：第一次 copy MLP 时，两个实例都持有
   同一个 cuBLAS handle，析构时都调 cublasDestroy → double free。修复：
   删除 MLP 的 copy ctor / assignment，显式定义 move 转移 handle 所有权。
3. **nlohmann/json 路径**：vendored 的 `json.hpp` 必须放在
   `third_party/nlohmann/nlohmann/json.hpp`，include 是 `<nlohmann/json.hpp>`。
   把头文件放在 `third_party/nlohmann/json.hpp` 但 include 写成 nlohmann:: 找不到。
4. **Qwen2 RMSNorm gamma 不是 1.0**：HF 训练后的模型里 `input_layernorm.weight`
   mean ≈ 0.25（不是初始值 1.0），最初的测试断言写 `fabs(mean - 1.0) < 0.05` 一直
   fail。改成 sanity 范围 [1e-3, 10] 后通过。
5. **mohomm 测试 print 写死 2D shape**：原始测试代码
   `printf("%zu,%zu", shape[0], shape[1] ?: 1)` 在 1D 张量（如 layernorm）打印成
   `[3584, 1]`，误以为 loader 把 shape 解释错。改成按 shape.size() 循环打印。
6. **nlohmann/json BF16 numpy 转换**：`t.astype(np.float32)` 在 numpy 上不支持
   bfloat16。改用 torch：`t.to(torch.float32)`。

### 文件清单（本周末）

```
src/
├── core/
│   ├── graph.{h,cpp}             # 新增 — 线性链计算图
│   ├── dtype_utils.h             # 新增 — f32→f16 bits
│   ├── tensor.{h,cpp}            # 扩展 — BF16 支持 + 深拷贝
│   └── allocator.{h,cpp}
├── model/
│   ├── model_config.{h,cpp}      # 新增
│   ├── safetensors_loader.{h,cpp} # 新增
│   └── qwen_model.{h,cpp}        # 新增
└── layers/
    ├── rmsnorm.{h,cpp}           # 改为 move-only
    ├── mlp.{h,cpp}               # 改为 move-only + init()
    ├── rope.{h,cpp}
    └── swiglu_kernel.cuh

third_party/nlohmann/nlohmann/json.hpp  # vendored
scripts/inspect_weights.py              # 新增 — torch 对照工具
tests/
├── test_model_config/                 # 新增
├── test_graph/                        # 新增
├── test_safetensors/                  # 新增 — 339 张量索引
└── test_qwen_model/                   # 新增 — 加载 14GB 模型
```
