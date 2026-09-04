"""Backend selection from ``[ai]`` config.

``get_backend()`` returns the backend for the configured provider.
``get_embed_backend()`` returns the same, except Anthropic (which has no
embeddings API) is transparently swapped for the configured ``embed_fallback``
provider. Both are ``lru_cache``d, so config is read once per process — call
:func:`reset_cache` in tests after changing config.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from ..config import load_config
from .base import AIBackend, AIUnavailableError


def _ai_config() -> Dict[str, Any]:
    return load_config().get("ai", {})


def _make(provider: str) -> AIBackend:
    provider = (provider or "").strip().lower()
    if provider == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend()
    if provider == "openai":
        from .openai import OpenAIBackend

        return OpenAIBackend()
    if provider == "openai-compatible":
        from .openai import OpenAIBackend

        return OpenAIBackend(compatible=True)
    if provider == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend()
    if provider == "gemini":
        from .gemini import GeminiBackend

        return GeminiBackend()
    raise AIUnavailableError(
        f"Unknown AI provider {provider!r}. "
        f"Set a valid [ai] provider in ~/.fixerr/config.toml "
        f"(ollama, openai, openai-compatible, anthropic, gemini)."
    )


@lru_cache(maxsize=1)
def get_backend() -> AIBackend:
    """Backend for the configured ``[ai] provider`` (used for generation)."""
    return _make(_ai_config().get("provider", "ollama"))


@lru_cache(maxsize=1)
def get_embed_backend() -> AIBackend:
    """Backend used for embeddings.

    Anthropic has no embeddings API, so when it is the active provider we return
    the ``[ai] embed_fallback`` backend instead (default: ollama).
    """
    ai = _ai_config()
    provider = ai.get("provider", "ollama")
    if (provider or "").strip().lower() == "anthropic":
        provider = ai.get("embed_fallback", "ollama")
    return _make(provider)


def reset_cache() -> None:
    """Clear the cached backends (config is re-read on next call). For tests."""
    get_backend.cache_clear()
    get_embed_backend.cache_clear()
