"""Multi-turn chat session with conversation history."""

from typing import List, Optional, Generator

from .config import InferenceConfig


class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"ChatMessage(role={self.role!r}, content={self.content!r})"


class ChatSession:
    """Multi-turn chat session with history management.

    Works identically offline and online: the backend formats the chat
    template (the local binary via `--raw`, the server via
    `/v1/chat/completions`), so this class holds only the message history.

    Example:
        engine = MiniInfer(model_path="...")          # or endpoint="..."
        chat = engine.chat(system_prompt="Be concise.")
        chat.send("Hello!")
        chat.send("Write a Python function to sort a list.")
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

    def _messages(self, user_message: str) -> List[dict]:
        msgs = [{"role": m.role, "content": m.content} for m in self.history]
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def send(self, message: str) -> ChatMessage:
        """Send a user message and return the assistant's response."""
        self.history.append(ChatMessage(role="user", content=message))
        text = self._engine._chat(self._messages(message), self._config)
        response = ChatMessage(role="assistant", content=text)
        self.history.append(response)
        return response

    def stream(self, message: str) -> Generator[str, None, None]:
        """Send a message and stream the response, yielding text chunks."""
        self.history.append(ChatMessage(role="user", content=message))
        collected = []
        for chunk in self._engine._chat_stream(self._messages(message), self._config):
            collected.append(chunk)
            yield chunk
        self.history.append(ChatMessage(role="assistant", content="".join(collected)))

    def clear(self):
        """Clear conversation history."""
        self.history.clear()

    def last_message(self) -> Optional[ChatMessage]:
        return self.history[-1] if self.history else None

    def __repr__(self):
        turns = sum(1 for m in self.history if m.role == "user")
        return f"ChatSession(turns={turns})"
