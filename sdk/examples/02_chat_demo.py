#!/usr/bin/env python3
"""
02_chat_demo.py — Interactive multi-turn chat with conversation history.

Usage:
    python sdk/examples/02_chat_demo.py --model /path/to/Qwen2.5-7B-Instruct
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini_infer_sdk import MiniInfer, InferenceConfig, SamplingConfig, SamplingMode


def code_review_chat(engine: MiniInfer):
    """Demonstrate: code review assistant that remembers context."""
    chat = engine.chat(
        system_prompt=(
            "You are a senior code reviewer. When reviewing code, point out "
            "bugs, performance issues, and suggest improvements. Be concise."
        )
    )

    print("=" * 60)
    print("  Code Review Chat Demo")
    print("=" * 60)

    code = '''def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)'''

    resp = chat.send(f"Review this code:\n```python\n{code}\n```")
    print(f"\n[Reviewer]: {resp.content}\n")

    resp = chat.send("Can you suggest an improved version?")
    print(f"[Reviewer]: {resp.content}\n")

    resp = chat.send("What is the time complexity of your improved version?")
    print(f"[Reviewer]: {resp.content}\n")

    print(f"Chat history: {len(chat.history)} messages")


def creative_writing_chat(engine: MiniInfer):
    """Demonstrate: creative writing with context preservation."""
    chat = engine.chat(
        system_prompt="You are a creative writing assistant. Help users write stories."
    )

    print("=" * 60)
    print("  Creative Writing Chat Demo")
    print("=" * 60)

    resp = chat.send("Start a short story about a robot learning to paint.")
    print(f"\n[Assistant]: {resp.content}\n")

    resp = chat.send("Continue the story with a twist.")
    print(f"[Assistant]: {resp.content}\n")

    resp = chat.send("Give the story a satisfying ending.")
    print(f"[Assistant]: {resp.content}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat demo")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--draft", default=None, help="Path to draft model")
    args = parser.parse_args()

    engine = MiniInfer(
        model_path=args.model,
        draft_path=args.draft,
    )

    code_review_chat(engine)
    print("\n" + "-" * 60 + "\n")
    creative_writing_chat(engine)


if __name__ == "__main__":
    main()