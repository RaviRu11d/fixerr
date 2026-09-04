"""fixerr — a local-first, provider-agnostic error/fix memory.

Public API:
    from fixerr import ErnestClient
    from fixerr.ai import get_backend, get_embed_backend
"""

from __future__ import annotations

from .client import ErnestClient

__all__ = ["ErnestClient", "__version__"]

__version__ = "0.2.0"
