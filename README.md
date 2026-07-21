# mini-infer

> 轻量级 C++/CUDA LLM 推理引擎：聚焦投机解码、PagedAttention、Prefix Caching 等工业级优化特性。

## 状态

**Week 3 / Phase 1: 模型加载 + 计算图** — 完成

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

## 测试覆盖（11/11 pass）

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

## 路线图（8 周）

| Week | 主题                                       | 状态 |
| ---- | ------------------------------------------ | ---- |
| 1    | 基础设施 / Tensor / Allocator              | ✓    |
| 2    | RMSNorm / RoPE / Softmax / SwiGLU / MLP GEMM | ✓    |
| 3    | safetensors loader + QwenModel + 计算图     | ✓    |
| 4    | 端到端 Qwen2.5 推理                        |      |
| 5    | PagedAttention                             |      |
| 6    | 连续批处理 + Benchmark                     |      |
| 7    | 投机解码 (draft + 验证)                    |      |
| 8    | KV Cache 回滚 + Prefix Cache + 收尾        |      |

## License

MIT