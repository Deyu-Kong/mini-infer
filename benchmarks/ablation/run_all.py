#!/usr/bin/env python3
"""
Run all ablation experiments (E0-E6) and generate summary report.

Usage:
    python benchmarks/ablation/run_all.py \
        --model /path/to/Qwen2.5-7B-Instruct \
        --draft /path/to/Qwen2.5-0.5B-Instruct \
        --output-dir benchmarks/ablation/results
"""

import argparse
import json
import subprocess
from pathlib import Path


def run_script(script: str, args: list) -> bool:
    """Run a Python script and return success status."""
    cmd = ["python3", script] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all ablation experiments")
    parser.add_argument("--model", required=True, help="Target model path")
    parser.add_argument("--draft", required=True, help="Draft model path")
    parser.add_argument("--binary", default="./build/mini_infer", help="Binary path")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per experiment")
    parser.add_argument("--output-dir", default="benchmarks/ablation/results",
                        help="Output directory for results")
    parser.add_argument("--skip", nargs="*", default=[], help="Experiments to skip (e0-e6)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [
        ("e0", "benchmarks/ablation/e0_baseline.py", [
            "--model", args.model,
            "--binary", args.binary,
            "--max-tokens", str(args.max_tokens),
            "--output", str(output_dir / "e0_baseline.json")
        ]),
        ("e1", "benchmarks/ablation/e1_paged.py", [
            "--model", args.model,
            "--binary", args.binary,
            "--max-tokens", str(args.max_tokens),
            "--output", str(output_dir / "e1_paged.json")
        ]),
        ("e2", "benchmarks/ablation/e2_batching.py", [
            "--model", args.model,
            "--binary", args.binary,
            "--max-tokens", str(args.max_tokens),
            "--output", str(output_dir / "e2_batching.json")
        ]),
        ("e3", "benchmarks/ablation/e3_spec_g4.py", [
            "--model", args.model,
            "--draft", args.draft,
            "--binary", args.binary,
            "--max-tokens", str(args.max_tokens),
            "--output", str(output_dir / "e3_spec_g4.json")
        ]),
        ("e4", "benchmarks/ablation/e4_spec_g8.py", [
            "--model", args.model,
            "--draft", args.draft,
            "--binary", args.binary,
            "--max-tokens", str(args.max_tokens),
            "--output", str(output_dir / "e4_spec_g8.json")
        ]),
        ("e5", "benchmarks/ablation/e5_prefix_cache.py", [
            "--model", args.model,
            "--binary", args.binary,
            "--max-tokens", str(args.max_tokens),
            "--output", str(output_dir / "e5_prefix_cache.json")
        ]),
        ("e6", "benchmarks/ablation/e6_tree_spec.py", [
            "--model", args.model,
            "--draft", args.draft,
            "--output", str(output_dir / "e6_tree_spec.json")
        ]),
    ]
    
    results = {}
    for exp_id, script, exp_args in experiments:
        if exp_id in args.skip:
            print(f"\n[SKIP] {exp_id.upper()}")
            continue
        
        success = run_script(script, exp_args)
        results[exp_id] = "success" if success else "failed"
    
    # Generate summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for exp_id, status in results.items():
        print(f"  {exp_id.upper()}: {status}")
    
    # Save summary
    summary = {
        "model": args.model,
        "draft_model": args.draft,
        "max_tokens": args.max_tokens,
        "experiments": results
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
