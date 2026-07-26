# mini-infer

> 轻量级 C++/CUDA LLM 推理引擎：聚焦投机解码、PagedAttention、Prefix Caching 等工业级优化特性。

## 状态

**Week 6 完成，进入 Week 7：投机解码 (draft + 验证)**

详细路线图见 [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)，周进度见 [`docs/WEEKLY_PROGRESS.md`](docs/WEEKLY_PROGRESS.md)。

## 目标模型

- Target: Qwen2.5-7B-Instruct (FP16)
- Draft : Qwen2.5-0.5B-Instruct (FP16, 投机解码用)

## 硬件

4 × RTX A6000 (48GB each), CUDA 12.x, sm_86。

## 快速构建

```bash
scripts/build.sh        # 配置 + 编译（自动设置环境）
scripts/run_tests.sh    # 运行 ctest 全套
```

也可手动：

```bash
export PATH=/data1/kdy/anaconda3/envs/vllm/bin:/usr/local/cuda-12.1/bin:/data1/tyh/miniconda3/bin:$PATH
export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 CUDAHOSTCXX=/usr/bin/g++-9
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j
cd build && ctest --output-on-failure
```

## Kernel 延迟（RTX A6000, FP16, 单 kernel 计时）

| Kernel | Shape                                | median (us) | p99 (us) |
| ------ | ------------------------------------ | ----------- | -------- |
| RMSNorm       | N=4, D=3584           |   6.14 |   8.19 |
| RoPE          | B=4, S=512, H=28, D=128 |  67.58 |  83.97 |
| Softmax       | N=57344, D=512          | 176.13 | 191.49 |
| SwiGLU        | N=2048, I=18944        | 348.16 | 349.18 |
| MLP (cuBLAS)  | B=4, H=512, I=1376     | 测一次端到端见 test_mlp |

> 完整测量见 `build/benchmarks/bench_kernels`。

## 测试覆盖（14/14 pass）

| 测试             | 对比对象                              |
| ---------------- | ------------------------------------- |
| tensor           | shape / h2d / d2h / add kernel        |
| allocator        | 256-byte 对齐 / 单调 / 溢出 / reset    |
| rmsnorm          | torch.rsqrt(x.pow(2).mean(...) + eps)*w |
| rope             | 数学公式（split last-dim + rotate）     |
| softmax          | torch.softmax(dim=-1)                  |
| swiglu           | silu(gate) * up                        |
| mlp              | 完整 SwiGLU MLP via cuBLAS GEMM       |
| model_config     | 4 个用例：valid / 缺字段 / GQA / 缺必填 |
| graph            | 空图 / 单节点 / Qwen 单 block / 打印    |
| safetensors      | 339 张量索引 / shape 校验 / BF16 round-trip |
| qwen_model       | 加载 14GB Qwen2.5-7B-Instruct，FP16 上 GPU |
| paged_attention  | prefill / decode 一致性 (FP16)         |
| request          | 状态机 / metrics / stop tokens         |
| prefix_cache     | Week 7-8 占位（no-op）                  |

## 路线图（8 周）

| Week | 主题                                       | 状态 |
| ---- | ------------------------------------------ | ---- |
| 1    | 基础设施 / Tensor / Allocator              | ✓    |
| 2    | RMSNorm / RoPE / Softmax / SwiGLU / MLP GEMM | ✓    |
| 3    | safetensors loader + QwenModel + 计算图     | ✓    |
| 4    | 端到端 Qwen2.5 推理                        | ✓    |
| 5    | PagedAttention                             | ✓    |
| 6    | 连续批处理 + Benchmark                     | ✓    |
| 7    | 投机解码 (draft + 验证)                    |      |
| 8    | KV Cache 回滚 + Prefix Cache + 收尾        |      |

## Week 6 Benchmark：连续批处理 vs 静态批处理

Qwen2.5-Coder-1.5B-Instruct + ShareGPT 1000 样本 + max_new_tokens=16:

| 模式 | 并发 | wall (ms) | throughput (tok/s) | speedup |
| ---- | ---- | --------- | ------------------ | ------- |
| static B=8 | 1000 | 75021 | 213 | 1.00x (baseline) |
| continuous | 1000 | 34992 | 457 | **2.14x** |

> 在所有请求同时到达的"最坏 case"下，连续批处理仍提供 2x+ 吞吐量提升。
> 在连续到达（arrival spread > 0）+ 高方差 prompt 长度的"理想 case"下，
> 连续批处理的理论上限接近 3-10x（vLLM 论文报告）。

折线图由 `scripts/plot_bench.py` 基于 `matplotlib` 生成：

![Throughput vs Concurrency](benchmarks/results/full/throughput_vs_concurrency_full.png)

完整 benchmark 报告：`benchmarks/results/full/comparison_full.csv` + `cont_*.md` / `static_*.md`。

## Week 7 预告：投机解码 (Speculative Decoding)

**目标**：集成 Draft 模型 (Qwen2.5-0.5B-Instruct)，实现并行验证 + 接受/拒绝采样，加速 decode 阶段。

### 计划交付物

- `src/speculative/draft_engine.{h,cpp}` — Draft 模型加载 + 快速推理
- `src/speculative/spec_decoder.{h,cpp}` — 投机解码主循环 (draft γ tokens → target verify → accept/reject)
- `src/kernels/verify_kernel.{cu,cuh}` — 并行验证 kernel (target 一次 forward γ+1 tokens)
- `src/kernels/accept_reject_kernel.{cu,cuh}` — 接受/拒绝采样 kernel
- `tests/test_spec_decoder/` — 投机解码单元测试 (greedy 必须与朴素解码一致)

### 关键技术点

1. **Draft 模型选择**：Qwen2.5-0.5B-Instruct 与 target 共享 tokenizer，是投机解码正确性前提
2. **并行验证**：target 模型一次 forward γ+1 tokens (draft 的 γ 个 + 1 个 bonus token)，而非逐个验证
3. **接受/拒绝采样**：基于 token 概率分布的 accept/reject，保证输出分布与朴素解码等价
4. **正确性验证**：固定种子下，投机解码输出必须与朴素解码完全一致 (greedy 模式)

### 预期收益

- **TPOT 降低**：decode 阶段每 token 延迟降低 2-4x (取决于 draft 模型接受率)
- **Throughput 提升**：在连续批处理基础上进一步叠加投机解码收益
- **消融实验 E3**：E2 + 投机解码 (γ=4)，测量 TPOT、加速比、接受率

详细技术设计见 [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) 第 4 节。

## License

MIT