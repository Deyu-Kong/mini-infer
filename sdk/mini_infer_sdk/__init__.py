"""
mini-infer SDK — Python interface for the mini-infer C++/CUDA engine.

The same API works in two modes:

  * Offline (local GPU): pass `model_path`; the SDK wraps the `mini_infer`
    binary via subprocess.
  * Online (persistent server): pass `endpoint` of a running
    `mini_infer serve` instance; the SDK talks to it over HTTP.

Quick Start — offline:

    from mini_infer_sdk import MiniInfer
    engine = MiniInfer(model_path="/path/to/Qwen2.5-7B-Instruct")
    print(engine.generate("Explain quantum computing.").text)

Quick Start — online:

    # start once: ./build/mini_infer serve --model /path/to/model --port 8000
    engine = MiniInfer(endpoint="http://localhost:8000")
    print(engine.generate("Explain quantum computing.").text)

Both modes support generate / stream / multi-turn chat / batch.
"""

from .config import InferenceConfig, SamplingConfig, SamplingMode
from .engine import MiniInfer
from .chat import ChatSession, ChatMessage
from .batch import BatchInfer
from .result import GenerationResult
from .server import MiniInferServer
from .backends import Backend, LocalBackend, RemoteBackend

__version__ = "0.2.0"
__all__ = [
    "MiniInfer", "MiniInferServer", "ChatSession", "ChatMessage", "BatchInfer",
    "GenerationResult", "InferenceConfig", "SamplingConfig", "SamplingMode",
    "Backend", "LocalBackend", "RemoteBackend",
]
