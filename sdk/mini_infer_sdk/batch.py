"""Batch inference for processing multiple prompts."""

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterator

from .config import InferenceConfig
from .engine import GenerationResult


@dataclass
class BatchResult:
    prompt: str
    result: GenerationResult
    index: int


@dataclass
class BatchStats:
    total_prompts: int
    total_generated_tokens: int
    total_elapsed: float
    tokens_per_sec: float
    results: List[BatchResult] = field(default_factory=list)

    def __repr__(self):
        return (f"BatchStats(prompts={self.total_prompts}, "
                f"tokens={self.total_generated_tokens}, "
                f"tps={self.tokens_per_sec:.1f})")


class BatchInfer:
    """Batch inference for processing multiple prompts efficiently.

    Supports both sequential and concurrent execution modes.

    Example:
        engine = MiniInfer(model_path="...")
        batch = engine.batch()

        prompts = [
            "Explain Python decorators.",
            "Write a binary search implementation.",
            "What is the time complexity of quicksort?",
        ]

        stats = batch.process(prompts)
        for r in stats.results:
            print(f"Q: {r.prompt[:50]}...")
            print(f"A: {r.result.text[:100]}...")
            print(f"  tps: {r.result.tokens_per_sec:.1f}")
            print()

        print(f"Overall: {stats.tokens_per_sec:.1f} tok/s")
    """

    def __init__(self, engine, config: Optional[InferenceConfig] = None):
        self._engine = engine
        self._config = config or engine.config

    def process(self, prompts: List[str], workers: int = 1) -> BatchStats:
        """Process a batch of prompts.

        Args:
            prompts: List of input prompts.
            workers: Number of concurrent workers (1 = sequential).

        Returns:
            BatchStats with results and aggregate metrics.
        """
        if workers <= 1:
            return self._process_sequential(prompts)

        return self._process_concurrent(prompts, workers)

    def _process_sequential(self, prompts: List[str]) -> BatchStats:
        results: List[BatchResult] = []
        total_elapsed = 0.0
        total_tokens = 0

        for i, prompt in enumerate(prompts):
            result = self._engine.generate(prompt, config=self._config)
            results.append(BatchResult(prompt=prompt, result=result, index=i))
            total_elapsed += result.elapsed_sec
            total_tokens += result.generated_tokens

        return BatchStats(
            total_prompts=len(prompts),
            total_generated_tokens=total_tokens,
            total_elapsed=total_elapsed,
            tokens_per_sec=total_tokens / total_elapsed if total_elapsed > 0 else 0,
            results=results,
        )

    def _process_concurrent(self, prompts: List[str], workers: int) -> BatchStats:
        results: List[BatchResult] = []
        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._engine.generate, prompt, self._config): (i, prompt)
                for i, prompt in enumerate(prompts)
            }

            for future in concurrent.futures.as_completed(futures):
                i, prompt = futures[future]
                try:
                    result = future.result()
                    results.append(BatchResult(prompt=prompt, result=result, index=i))
                except Exception as e:
                    print(f"[BatchInfer] Error processing prompt {i}: {e}")

        total_elapsed = time.perf_counter() - t0
        results.sort(key=lambda r: r.index)
        total_tokens = sum(r.result.generated_tokens for r in results)

        return BatchStats(
            total_prompts=len(prompts),
            total_generated_tokens=total_tokens,
            total_elapsed=total_elapsed,
            tokens_per_sec=total_tokens / total_elapsed if total_elapsed > 0 else 0,
            results=results,
        )

    def stream(self, prompts: List[str]) -> Iterator[BatchResult]:
        """Process prompts sequentially, yielding results as they complete.

        Args:
            prompts: List of input prompts.

        Yields:
            BatchResult for each completed prompt.
        """
        for i, prompt in enumerate(prompts):
            result = self._engine.generate(prompt, config=self._config)
            yield BatchResult(prompt=prompt, result=result, index=i)