# mini-infer 周进度记录

> 每周交付物 + 验收记录 + 踩坑记录

---

## Week 1 — 基础设施（2026-07-20 ~ 07-26）

**目标**：搭建 CUDA 开发环境，建立项目骨架，跑通最小 CUDA kernel。

### 交付清单

- [x] Git 仓库初始化（git init + .gitignore + LICENSE + README）
- [x] CMakeLists.txt（CUDAToolkit + C++17，编译 hello kernel 通过）
- [x] `src/core/tensor.h/cpp` — Tensor 抽象（FP16/FP32，d2d/h2d，stride）
- [x] `src/core/allocator.h/cpp` — BumpAllocator（256B 对齐）
- [x] `examples/00_hello_cuda` — kernel launch + GPU info 输出
- [x] `tests/test_tensor` — Tensor 往返 + add kernel 对拍

### 验收命令

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/examples/00_hello_cuda        # 输出 GPU 信息
./build/tests/test_tensor             # Tensor + AddKernel 验证
```

### 技术决策

- **CUDA 版本**：使用本地 `/usr/local/cuda-12.1/`（驱动 12.2 向后兼容 12.1 toolkit）。
- **GPU arch**：A6000 = sm_86，CMake 中显式 `CMAKE_CUDA_ARCHITECTURES 86`。
- **Tensor stride**：用 `std::vector<int64_t>` 存，缺省为 row-major contiguous。
- **Allocator**：256-byte 对齐（cuBLAS 要求），先 bump，后续扩展为 block pool。

### 踩坑

- _（实现后补充）_

---

## Week 2 — 基础算子（占位）

_TBD_