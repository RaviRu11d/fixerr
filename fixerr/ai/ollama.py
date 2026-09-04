"""Local Ollama backend — the zero-dependency default.

Talks to a local Ollama daemon over its HTTP API using ``httpx`` (a base
dependency), so no provider SDK is needed. Any connection problem degrades to
:class:`AIUnavailableError`; callers (search/surface) then fall back to plain
text matching.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..config import load_config
from .base import AIBackend, AIUnavailableError


class OllamaBackend(AIBackend):
    def __init__(self) -> None:
        cfg = load_config().get("ai", {}).get("ollama", {})
        env_host = os.environ.get("OLLAMA_HOST")
        self._host = (env_host or cfg.get("host", "http://localhost:11434")).rstrip("/")
        self._embed_model = cfg.get("embed_model", "nomic-embed-text")
        self._gen_model = cfg.get("gen_model", "llama3")
        # Generation is inherently slower than embedding (especially on
        # larger/reasoning local models — a "ping" can be fast while a real
        # multi-paragraph prompt takes well over 30s) and its callers
        # (`fixerr explain`, `fixerr run`) already run detached/backgrounded,
        # so there's no UX cost to a generous timeout here.
        self._embed_timeout = float(cfg.get("embed_timeout", 30))
        self._gen_timeout = float(cfg.get("gen_timeout", 180))

    def _post(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a base dep
            raise AIUnavailableError(
                "httpx package not installed. Run: pip install fixerr"
            ) from exc
        try:
            resp = httpx.post(f"{self._host}{path}", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise AIUnavailableError(
                f"Ollama at {self._host} didn't respond within {timeout:.0f}s. It may be "
                f"reachable but overloaded/slow for this model — try raising "
                f"[ai.ollama] gen_timeout in ~/.fixerr/config.toml."
            ) from exc
        except Exception as exc:  # httpx.HTTPError, ConnectError, JSON errors, etc.
            raise AIUnavailableError(
                f"Ollama unreachable at {self._host} ({exc.__class__.__name__}). "
                f"Is it running? Try: ollama serve"
            ) from exc

    def embed(self, text: str) -> List[float]:
        data = self._post(
            "/api/embeddings", {"model": self._embed_model, "prompt": text}, self._embed_timeout
        )
        vector = data.get("embedding")
        if not isinstance(vector, list):
            raise AIUnavailableError("Ollama returned no embedding.")
        return [float(x) for x in vector]

    def generate(self, prompt: str) -> str:
        data = self._post(
            "/api/generate",
            {"model": self._gen_model, "prompt": prompt, "stream": False},
            self._gen_timeout,
        )
        return str(data.get("response", ""))
