"""Anthropic backend — generation only.

Anthropic exposes no embeddings API, so :meth:`embed` raises
``NotImplementedError`` and the factory routes embeddings to the configured
``embed_fallback`` provider instead (see :func:`fixerr.ai.factory.get_embed_backend`).
The SDK is imported lazily so fixerr runs without it until this provider is
selected.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..config import load_config
from .base import AIBackend, AIUnavailableError


class AnthropicBackend(AIBackend):
    def __init__(self) -> None:
        cfg: Dict[str, Any] = load_config().get("ai", {}).get("anthropic", {})
        self._api_key_env = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        self._gen_model = cfg.get("gen_model", "claude-haiku-4-5")

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError(
            "Anthropic has no embeddings API. Set embed_fallback in config."
        )

    def generate(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise AIUnavailableError(
                "anthropic package not installed. Run: pip install fixerr[anthropic]"
            ) from exc

        api_key = os.environ.get(self._api_key_env or "")
        if not api_key:
            raise AIUnavailableError(
                f"{self._api_key_env} is not set. "
                f"Export it or set [ai.anthropic] api_key_env in ~/.fixerr/config.toml."
            )

        try:
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=self._gen_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as exc:
            raise AIUnavailableError(
                f"Anthropic generation failed ({exc.__class__.__name__}): {exc}"
            ) from exc
