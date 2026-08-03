"""Backend interface shared by local (subprocess) and remote (HTTP) backends."""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List

from ..config import InferenceConfig
from ..result import GenerationResult


class Backend(ABC):
    """A generation backend.

    `generate` / `stream` operate on a raw single-turn prompt; `chat` /
    `chat_stream` operate on a list of `{role, content}` messages (the
    backend is responsible for chat-template formatting). This split keeps
    `ChatSession` free of tokenizer details and works identically offline
    (subprocess wraps the prompt in ChatML or uses `--raw`) and online
    (server formats via `/v1/chat/completions`).
    """

    config: InferenceConfig

    @abstractmethod
    def generate(self, prompt: str, config: InferenceConfig) -> GenerationResult:
        ...

    @abstractmethod
    def stream(self, prompt: str, config: InferenceConfig) -> Iterator[str]:
        ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]],
             config: InferenceConfig) -> str:
        ...

    @abstractmethod
    def chat_stream(self, messages: List[Dict[str, str]],
                    config: InferenceConfig) -> Iterator[str]:
        ...

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        ...

    @abstractmethod
    def detokenize(self, ids: List[int]) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    def is_remote(self) -> bool:
        return False
