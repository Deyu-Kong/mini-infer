# mini-infer

> 轻量级 C++/CUDA LLM 推理引擎：聚焦投机解码、PagedAttention、Prefix Caching 等工业级优化特性。

## 状态

**Week 1 / Phase 1: 基础设施** — 进行中

详细路线图见 [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)，周进度见 [`docs/WEEKLY_PROGRESS.md`](docs/WEEKLY_PROGRESS.md)。

## 目标模型

- Target: Qwen2.5-7B-Instruct (FP16)
- Draft : Qwen2.5-0.5B-Instruct (FP16, 投机解码用)

## 硬件

4 × RTX A6000 (48GB each), CUDA 12.x, sm_86。

## 快速构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/examples/00_hello_cuda
./build/tests/test_tensor
```

## 路线图（8 周）

| Week | 主题                        |
| ---- | --------------------------- |
| 1    | 基础设施 / Tensor / Allocator |
| 2    | RMSNorm / RoPE / Softmax kernel |
| 3    | safetensors loader + 计算图 |
| 4    | 端到端 Qwen2.5 推理         |
| 5    | PagedAttention              |
| 6    | 连续批处理 + Benchmark      |
| 7    | 投机解码 (draft + 验证)     |
| 8    | KV Cache 回滚 + Prefix Cache + 收尾 |

## License

MIT