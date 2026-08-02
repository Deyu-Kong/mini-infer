#!/usr/bin/env python3
"""
01_quickstart.py — Minimal example to get started with mini-infer SDK.

Usage:
    python sdk/examples/01_quickstart.py --model /path/to/Qwen2.5-7B-Instruct
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini_infer_sdk import MiniInfer, InferenceConfig, SamplingConfig, SamplingMode


def main():
    parser = argparse.ArgumentParser(description="mini-infer SDK Quick Start")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--draft", default=None, help="Path to draft model (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("  mini-infer SDK — Quick Start")
    print("=" * 60)

    # Step 1: Initialize the engine (binary auto-discovered)
    print("\n[1/3] Loading model...")
    engine = MiniInfer(
        model_path=args.model,
        draft_path=args.draft,
    )
    print(f"  Engine: {engine}")

    # Step 2: Generate a single response
    print("\n[2/3] Generating response...")
    config = InferenceConfig(
        max_new_tokens=128,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
    )
    result = engine.generate("Explain the transformer architecture in 3 sentences.", config=config)
    print(f"  Result: {result}")
    print(f"  Output: {result.text[:200]}...")

    # Step 3: Multi-turn chat
    print("\n[3/3] Multi-turn chat...")
    chat = engine.chat(system_prompt="You are a concise AI assistant. Keep answers short.")
    resp1 = chat.send("What is the capital of France?")
    print(f"  User: What is the capital of France?")
    print(f"  Assistant: {resp1.content[:100]}")

    resp2 = chat.send("What is its population?")
    print(f"  User: What is its population?")
    print(f"  Assistant: {resp2.content[:100]}")

    print(f"\n  Chat history: {len(chat.history)} messages")

    print("\n" + "=" * 60)
    print("  Done! Try the other examples in sdk/examples/")
    print("=" * 60)


if __name__ == "__main__":
    main()