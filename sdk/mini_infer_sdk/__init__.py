"""
mini-infer SDK — Python interface for the mini-infer C++/CUDA inference engine.

Quick Start:
    from mini_infer_sdk import MiniInfer

    engine = MiniInfer(
        model_path="/path/to/Qwen2.5-7B-Instruct",
        draft_path="/path/to/Qwen2.5-Coder-1.5B-Instruct",  # optional
    )

    # Single prompt
    result = engine.generate("Explain quantum computing in one sentence.")
    print(result.text)

    # Multi-turn chat
    chat = engine.chat()
    chat.send("Hello!")
    chat.send("Write a Python function to reverse a string.")
    for resp in chat.history:
        print(f"[{resp.role}]: {resp.content}")
"""

from .engine import MiniInfer
from .chat import ChatSession, ChatMessage
from .batch import BatchInfer
from .config import InferenceConfig, SamplingConfig, SamplingMode

__version__ = "0.1.0"
__all__ = ["MiniInfer", "ChatSession", "ChatMessage", "BatchInfer", "InferenceConfig", "SamplingConfig", "SamplingMode"]