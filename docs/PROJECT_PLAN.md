# mini-infer 项目规划

> 一款轻量级 C++/CUDA LLM 推理引擎。核心目标：在紧凑代码量内实现投机解码、PagedAttention、Prefix Caching 等工业级推理优化特性，并做严谨的消融实验量化每个特性的贡献。

---

## 1. 项目定调

| 决策点     | 选择                          | 理由                                                         |
| ---------- | ----------------------------- | ------------------------------------------------------------ |
| 主语言     | C++17 + CUDA 12.x             | 体现底层工程能力                                             |
| GEMM       | 调用 cuBLAS                   | 其他算子全部自己写                                           |
| 模型格式   | HuggingFace safetensors       | 主流格式，解析简单                                           |
| Target     | Qwen2.5-7B-Instruct           | FP16 推理 ~14GB，单卡 48GB 充裕                              |
| Draft      | Qwen2.5-0.5B-Instruct         | 与 target 共享 tokenizer，是投机解码正确性前提                |
| 构建       | CMake + find_package(CUDA)    | 标准做法                                                     |
| 性能分析   | Nsight Compute / Nsight Systems | kernel 级 + 系统级 profiling                               |

## 2. 硬件环境

- GPU: 4 × NVIDIA RTX A6000 (Ampere, sm_86), 每张 48GB ≈ 192GB 总量
- 驱动: NVIDIA 535.129.03, CUDA 12.2 compatible
- 本地 Toolkit: `/usr/local/cuda-12.1/` (使用 12.1 兼容 12.2 特性集)
- 验证: 开发用 GPU0, 其余三张可同时跑对照实验

## 3. 代码结构

```
mini-infer/
├── CMakeLists.txt
├── src/
│   ├── core/        # Tensor, Allocator, Graph, Engine
│   ├── layers/      # Attention, MLP, RMSNorm, RoPE
│   ├── kernels/     # *.cu 自实现 kernel
│   ├── model/       # safetensors loader, qwen model
│   ├── scheduler/   # 调度器, prefix cache, request
│   └── speculative/ # draft engine, spec decoder, tree spec
├── tests/           # 单元测试（与 PyTorch 对拍）
├── benchmarks/      # benchmark 脚本
├── examples/        # 可运行示例
├── docs/            # 设计文档 / 路线图
└── scripts/         # 工具脚本
```

## 4. 八周路线图

| Week | 主题                | 关键交付物                                                  |
| ---- | ------------------- | ----------------------------------------------------------- |
| W1   | 基础设施            | CMake 骨架, Tensor, BumpAllocator, hello kernel            |
| W2   | 基础算子            | RMSNorm / RoPE / Softmax kernel + PyTorch 对拍测试          |
| W3   | 模型加载            | safetensors 解析, config 解析, 计算图构建                  |
| W4   | 端到端推理          | KV Cache (朴素连续版), tokenizer, sampling, 跑通 Qwen2.5    |
| W5   | PagedAttention      | Block pool + paged attention CUDA kernel                    |
| W6   | 连续批处理 + Bench  | Scheduler + 变长 padding + Benchmark 框架                   |
| W7   | 投机解码 (Part 1)   | Draft 模型集成, 并行验证, 接受/拒绝采样 kernel             |
| W8   | 投机解码 (Part 2)   | KV Cache 回滚, (可选) 树形投机 EAGLE-2                      |
| W7+  | Prefix Cache        | Radix Trie, block hash, LRU 淘汰, copy-on-write             |
| W8   | 收尾                | 消融实验 E0~E6, 技术博客, GitHub 开源                       |

## 5. 消融实验设计

| 实验 | 配置                              | 测量指标                       |
| ---- | --------------------------------- | ------------------------------ |
| E0   | 基线朴素自回归                    | TTFT, TPOT, Throughput         |
| E1   | E0 + PagedAttention               | 显存利用率, 最大并发数         |
| E2   | E1 + 连续批处理                   | Throughput 提升倍数            |
| E3   | E2 + 投机解码 (γ=4)               | TPOT, 加速比, 接受率           |
| E4   | E3 + 投机解码 (γ=8)               | γ 对加速比的影响               |
| E5   | E3 + Prefix Caching               | 共享前缀场景 TTFT 降低         |
| E6   | E3 + 树形投机（若实现）           | 与 vanilla 投机对比            |

## 6. 关键避坑

1. **正确性先于性能**：前 4 周不碰优化。每个算子必须与 PyTorch 对拍。
2. **投机解码分布等价**：固定种子，朴素 vs 投机输出必须完全一致 (greedy)。
3. **不要试图优化所有东西**：GEMM/Tokenizer 调库，精力集中在 PagedAttention / 投机 / Prefix。
4. **尽早建立 Benchmark**：W6 搭好框架，后面每加一个特性立刻测。