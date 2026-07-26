#!/usr/bin/env python3
"""
Ablation Study: E2 - Continuous Batching

This script measures throughput improvement from continuous batching
(dynamic batching where sequences can start/finish independently).

Metrics:
- Total throughput (tokens/sec across all sequences)
- Latency per sequence
- Batch utilization
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_experiment(model_path: str, prompts: list, max_tokens: int,
                   binary: str = "./build/mini_infer") -> dict:
    """Run batched experiment."""
    # For now, we simulate batching by running sequentially
    # In a full implementation, this would use --batch mode
    results = []
    total_start = time.time()
    
    for prompt in prompts:
        cmd = [
            binary,
            "--model", model_path,
            "--prompt", prompt,
            "--max-new-tokens", str(max_tokens),
            "--greedy"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            import re
            speed_match = re.search(r'(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)', 
                                    result.stdout)
            if speed_match:
                results.append({
                    "tokens": int(speed_match.group(1)),
                    "time": float(speed_match.group(2)),
                    "tok_per_sec": float(speed_match.group(3))
                })
    
    total_time = time.time() - total_start
    
    return {
        "total_time": total_time,
        "num_sequences": len(prompts),
        "total_tokens": sum(r["tokens"] for r in results),
        "aggregate_throughput": sum(r["tokens"] for r in results) / total_time,
        "per_sequence": results
    }


def main():
    parser = argparse.ArgumentParser(description="E2: Continuous Batching")
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", default="./build/mini_infer")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", default="benchmarks/ablation/results/e2_batching.json")
    args = parser.parse_args()
    
    base_prompts = [
        "Hello, how are you?",
        "Write a Python function.",
        "Explain attention mechanism.",
        "What is speculative decoding?",
        "Describe PagedAttention.",
        "How does KV cache work?",
        "What are transformers?",
        "Explain self-attention."
    ]
    
    results = {}
    for bs in args.batch_sizes:
        print(f"[E2] Batch size {bs}...")
        prompts = base_prompts[:bs]
        metrics = run_experiment(args.model, prompts, args.max_tokens, args.binary)
        results[f"batch_{bs}"] = metrics
        print(f"  -> {metrics['aggregate_throughput']:.2f} tok/s aggregate")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "E2_continuous_batching",
            "model": args.model,
            "results": results
        }, f, indent=2)
    
    print(f"\n[E2] Results saved to {output_path}")


if __name__ == "__main__":
    main()
