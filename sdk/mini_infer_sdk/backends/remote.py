"""Remote backend — HTTP client to a running `mini_infer serve` instance."""

import json as _json
from typing import Dict, Iterator, List, Optional

import requests

from ..config import InferenceConfig, SamplingMode
from ..result import GenerationResult
from .base import Backend


class RemoteBackend(Backend):
    """Online backend: talks to a persistent mini-infer HTTP server.

    The server (started with `mini_infer serve`) loads the model once into
    GPU memory, so every request reuses the loaded weights — no per-call
    reload, unlike :class:`LocalBackend`. Exposes the same Backend API, so
    :class:`mini_infer_sdk.MiniInfer` behaves identically offline and online.
    """

    def __init__(self, endpoint: str,
                 config: Optional[InferenceConfig] = None,
                 timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.config = config or InferenceConfig()
        self.timeout = timeout
        self._model_name: Optional[str] = None

    @property
    def is_remote(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            try:
                r = requests.get(f"{self.endpoint}/health", timeout=self.timeout)
                r.raise_for_status()
                self._model_name = r.json().get("model", "remote")
            except Exception:
                self._model_name = "remote"
        return self._model_name

    def _merge(self, config: Optional[InferenceConfig]) -> InferenceConfig:
        return config or self.config

    def _sampling_body(self, cfg: InferenceConfig) -> Dict:
        return {
            "temperature": cfg.sampling.temperature,
            "top_p": cfg.sampling.top_p,
            "greedy": cfg.sampling.mode == SamplingMode.GREEDY,
            "seed": cfg.sampling.seed,
        }

    # -- Backend API --------------------------------------------------------
    def generate(self, prompt: str, config: Optional[InferenceConfig] = None) -> GenerationResult:
        cfg = self._merge(config)
        body = {
            "prompt": prompt,
            "max_new_tokens": cfg.max_new_tokens,
            **self._sampling_body(cfg),
        }
        r = requests.post(f"{self.endpoint}/generate", json=body, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return GenerationResult(
            text=d.get("text", ""),
            prompt_tokens=d.get("prompt_tokens", 0),
            generated_tokens=d.get("generated_tokens"),
            elapsed_sec=d.get("elapsed", 0.0),
            tokens_per_sec=d.get("tokens_per_sec"),
            finish_reason=d.get("finish_reason"),
        )

    def chat(self, messages: List[Dict[str, str]],
             config: Optional[InferenceConfig] = None) -> str:
        cfg = self._merge(config)
        body = {
            "messages": messages,
            "max_tokens": cfg.max_new_tokens,
            **self._sampling_body(cfg),
            "stream": False,
        }
        r = requests.post(f"{self.endpoint}/v1/chat/completions",
                          json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0]["message"]["content"]

    def chat_stream(self, messages: List[Dict[str, str]],
                    config: Optional[InferenceConfig] = None) -> Iterator[str]:
        cfg = self._merge(config)
        body = {
            "messages": messages,
            "max_tokens": cfg.max_new_tokens,
            **self._sampling_body(cfg),
            "stream": True,
        }
        with requests.post(f"{self.endpoint}/v1/chat/completions",
                           json=body, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    ev = _json.loads(payload)
                except _json.JSONDecodeError:
                    continue
                choices = ev.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    yield delta["content"]

    def stream(self, prompt: str, config: Optional[InferenceConfig] = None) -> Iterator[str]:
        # Single-turn stream == chat stream with one user message (server
        # applies the chat template), matching LocalBackend.stream semantics.
        yield from self.chat_stream([{"role": "user", "content": prompt}],
                                    config)

    def tokenize(self, text: str) -> List[int]:
        r = requests.post(f"{self.endpoint}/tokenize",
                          json={"text": text}, timeout=self.timeout)
        r.raise_for_status()
        return list(r.json().get("ids", []))

    def detokenize(self, ids: List[int]) -> str:
        r = requests.post(f"{self.endpoint}/detokenize",
                          json={"ids": list(ids)}, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("text", "")
