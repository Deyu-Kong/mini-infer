"""MiniInfer — unified facade over offline and online backends.

Usage — offline (local GPU, one process per call):

    from mini_infer_sdk import MiniInfer
    engine = MiniInfer(model_path="/path/to/Qwen2.5-7B-Instruct")
    result = engine.generate("Explain transformers.")

Usage — online (persistent in-memory server):

    # terminal 1: start the server once
    #   ./build/mini_infer serve --model /path/to/model --port 8000
    engine = MiniInfer(endpoint="http://localhost:8000")
    result = engine.generate("Explain transformers.")

`generate`, `stream`, `chat`, `batch`, `tokenize` all behave the same in
both modes; the backend is selected from whether `model_path` (offline) or
`endpoint` (online) is supplied.
"""

from pathlib import Path
import sys
from typing import List, Optional

from .backends import Backend, LocalBackend, RemoteBackend
from .config import InferenceConfig
from .result import GenerationResult


class MiniInfer:
    """Python interface for the mini-infer engine.

    Args (offline):
        model_path:  HuggingFace model directory.
        draft_path:  Optional draft model for speculative decoding.
        binary:      Path to mini_infer binary (auto-discovered if None).
        python_path: Python executable for the engine's tokenizer. Defaults
                     to ``sys.executable`` (the interpreter running the SDK,
                     which already has the `tokenizers` dependency) so you
                     normally never set this.
    Args (online):
        endpoint:    URL of a running `mini_infer serve` instance
                     (or a :class:`MiniInferServer` you started).
    Common:
        config:      Default inference configuration.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        draft_path: Optional[str] = None,
        binary: Optional[str] = None,
        python_path: Optional[str] = None,
        endpoint: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ):
        python_path = python_path or sys.executable
        self.config = config or InferenceConfig(python_path=python_path)
        self.config.draft_path = draft_path

        if endpoint:
            self.backend: Backend = RemoteBackend(endpoint, config=self.config)
        elif model_path:
            self.backend = LocalBackend(
                model_path=model_path,
                draft_path=draft_path,
                binary=binary,
                python_path=python_path,
                config=self.config,
            )
        else:
            raise ValueError(
                "Provide either model_path (offline) or endpoint (online)."
            )

    # -- Properties ---------------------------------------------------------
    @property
    def is_remote(self) -> bool:
        return self.backend.is_remote

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    # -- Core API -----------------------------------------------------------
    def generate(self, prompt: str,
                 config: Optional[InferenceConfig] = None) -> GenerationResult:
        return self.backend.generate(prompt, config or self.config)

    def stream(self, prompt: str, config: Optional[InferenceConfig] = None):
        yield from self.backend.stream(prompt, config or self.config)

    def tokenize(self, text: str) -> List[int]:
        return self.backend.tokenize(text)

    def detokenize(self, ids: List[int]) -> str:
        return self.backend.detokenize(ids)

    # -- Chat / batch (delegate to backend so both modes work uniformly) ----
    def _chat(self, messages, config) -> str:
        return self.backend.chat(messages, config or self.config)

    def _chat_stream(self, messages, config):
        yield from self.backend.chat_stream(messages, config or self.config)

    def chat(self, system_prompt: Optional[str] = None,
             config: Optional[InferenceConfig] = None):
        from .chat import ChatSession
        return ChatSession(self, system_prompt=system_prompt, config=config)

    def batch(self, config: Optional[InferenceConfig] = None):
        from .batch import BatchInfer
        return BatchInfer(self, config=config or self.config)

    def __repr__(self):
        kind = "remote" if self.is_remote else "local"
        name = self.model_name
        return f"MiniInfer({kind}, model={name})"
