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

## Week 2 — 基础算子（占位）

_TBD_