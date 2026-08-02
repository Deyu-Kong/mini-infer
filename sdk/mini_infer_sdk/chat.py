"""Multi-turn chat session with conversation history."""

from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field

from .config import InferenceConfig, SamplingConfig, SamplingMode


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatSession:
    """Multi-turn chat session with history management.

    Example:
        engine = MiniInfer(model_path="...")
        chat = engine.chat(system_prompt="You are a helpful assistant.")

        chat.send("Hello!")
        chat.send("Write a Python function to sort a list.")
        chat.send("Now add type hints.")

        for msg in chat.history:
            print(f"[{msg.role}]: {msg.content}")
    """

    def __init__(self, engine, system_prompt: Optional[str] = None,
                 config: Optional[InferenceConfig] = None):
        self._engine = engine
        self._config = config or engine.config
        self.history: List[ChatMessage] = []

        if system_prompt:
            self.history.append(ChatMessage(role="system", content=system_prompt))

    def _build_prompt(self, user_message: str) -> str:
        has_im_tokens = self._engine._tokenizer.token_to_id("<|im_start|>") is not None

        if has_im_tokens:
            parts = []
            if self.history:
                for msg in self.history:
                    parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
            parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)
        else:
            parts = []
            for msg in self.history:
                if msg.role == "system":
                    parts.append(f"System: {msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"Assistant: {msg.content}")
                elif msg.role == "user":
                    parts.append(f"User: {msg.content}")
            parts.append(f"User: {user_message}")
            parts.append("Assistant: ")
            return "\n".join(parts)

    def send(self, message: str) -> "ChatMessage":
        """Send a message and get the assistant's response.

        Args:
            message: User's input message.

        Returns:
            ChatMessage with role='assistant' and the generated content.
        """
        self.history.append(ChatMessage(role="user", content=message))

        prompt = self._build_prompt(message)
        result = self._engine.generate(prompt, config=self._config)

        response = ChatMessage(role="assistant", content=result.text)
        self.history.append(response)

        return response

    def stream(self, message: str) -> Generator[str, None, None]:
        """Send a message and stream the response.

        Args:
            message: User's input message.

        Yields:
            String chunks of the assistant's response.
        """
        self.history.append(ChatMessage(role="user", content=message))
        prompt = self._build_prompt(message)

        full_response = []
        for chunk in self._engine.stream(prompt, config=self._config):
            full_response.append(chunk)
            yield chunk

        response = ChatMessage(role="assistant", content="".join(full_response))
        self.history.append(response)

    def clear(self):
        """Clear conversation history."""
        self.history.clear()

    def last_message(self) -> Optional[ChatMessage]:
        """Get the last message in the conversation."""
        return self.history[-1] if self.history else None

    def __repr__(self):
        return f"ChatSession(turns={len([m for m in self.history if m.role == 'user'])})"