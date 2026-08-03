"""Generation result shared by all backends."""

from typing import List, Optional


class GenerationResult:
    """Result of a generation call.

    Attributes:
        text:             Generated text.
        tokens:           Full token list (prompt + generated) when available.
        prompt_tokens:    Number of prompt tokens.
        generated_tokens: Number of tokens generated.
        elapsed_sec:      Pure inference time (excludes model loading).
        wall_sec:         Wall-clock time including model loading / process spawn.
        tokens_per_sec:   Pure inference throughput (tok/s).
        accept_rate:      Speculative decoding acceptance rate (or None).
        finish_reason:    "stop" | "length" (online path).
    """

    def __init__(self, text: str, tokens: Optional[List[int]] = None,
                 prompt_tokens: int = 0,
                 generated_tokens: Optional[int] = None,
                 elapsed_sec: float = 0.0,
                 accept_rate: Optional[float] = None,
                 tokens_per_sec: Optional[float] = None,
                 wall_sec: Optional[float] = None,
                 finish_reason: Optional[str] = None):
        self.text = text
        self.tokens = tokens or []
        self.prompt_tokens = prompt_tokens
        # Prefer an explicit count (online servers report it directly);
        # otherwise derive from the token list (offline path).
        if generated_tokens is not None:
            self.generated_tokens = generated_tokens
        else:
            self.generated_tokens = (len(tokens) - prompt_tokens) if tokens else 0
        self.elapsed_sec = elapsed_sec
        self.wall_sec = wall_sec if wall_sec is not None else elapsed_sec
        if tokens_per_sec is not None:
            self.tokens_per_sec = tokens_per_sec
        else:
            self.tokens_per_sec = (
                self.generated_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
            )
        self.accept_rate = accept_rate
        self.finish_reason = finish_reason

    def __repr__(self):
        extra = f", accept_rate={self.accept_rate:.0%}" if self.accept_rate else ""
        return (f"GenerationResult(generated={self.generated_tokens}, "
                f"tps={self.tokens_per_sec:.1f}{extra})")
