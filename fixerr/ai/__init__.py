"""fixerr AI layer — provider-agnostic ``embed`` / ``generate``.

The rest of the codebase imports only from here (``from fixerr.ai import
get_backend`` / ``get_embed_backend``); it never touches a concrete provider.
This package supersedes the original single-file ``fixerr/ai.py`` and keeps the
same import path, so existing call sites do not change.
"""

from __future__ import annotations

from .base import AIBackend, AIUnavailableError
from .factory import get_backend, get_embed_backend, reset_cache

__all__ = [
    "AIBackend",
    "AIUnavailableError",
    "get_backend",
    "get_embed_backend",
    "reset_cache",
]
