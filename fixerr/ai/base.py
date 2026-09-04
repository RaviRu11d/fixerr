"""Provider-agnostic AI backend contract.

The rest of fixerr depends only on :class:`AIBackend` (via
``get_backend`` / ``get_embed_backend`` in :mod:`fixerr.ai.factory`) — it never
imports a concrete provider. Concrete backends live in sibling modules and must
translate *every* provider-specific failure (network error, missing SDK,
missing/invalid credentials) into :class:`AIUnavailableError`, so callers can
degrade gracefully with a single ``except``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class AIUnavailableError(RuntimeError):
    """The configured provider is unreachable, unconfigured, or missing its SDK.

    Backends raise this instead of leaking raw HTTP / SDK exceptions so that a
    single ``except AIUnavailableError`` at the call site covers every provider.
    """


class AIBackend(ABC):
    """Two operations, one interface, many providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for ``text``.

        Raises :class:`AIUnavailableError` when the provider is unreachable or
        unconfigured. May raise :class:`NotImplementedError` for providers with
        no embeddings API (e.g. Anthropic) — the factory routes around those.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return generated text for ``prompt``.

        Raises :class:`AIUnavailableError` when the provider is unreachable or
        unconfigured.
        """
        raise NotImplementedError
