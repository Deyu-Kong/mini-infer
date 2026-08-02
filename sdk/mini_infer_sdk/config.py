"""Configuration dataclasses for mini-infer SDK."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SamplingMode(str, Enum):
    GREEDY = "greedy"
    TOP_P = "top_p"


@dataclass
class SamplingConfig:
    mode: SamplingMode = SamplingMode.GREEDY
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 42


@dataclass
class InferenceConfig:
    max_new_tokens: int = 256
    max_seq_len: int = 2048
    device: int = 0
    paged: bool = True
    gamma: int = 4
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    draft_path: Optional[str] = None
    python_path: str = "python3"