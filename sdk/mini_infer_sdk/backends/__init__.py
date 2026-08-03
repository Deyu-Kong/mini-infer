"""Backend abstraction: the same MiniInfer API works offline and online."""

from .base import Backend
from .local import LocalBackend
from .remote import RemoteBackend

__all__ = ["Backend", "LocalBackend", "RemoteBackend"]
