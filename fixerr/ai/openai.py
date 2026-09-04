"""OpenAI backend (and OpenAI-compatible servers).

The same backend serves both ``[ai.openai]`` and ``[ai.openai-compatible]``:
the only difference is that the compatible variant sets ``base_url`` on the
client, pointing it at a local or third-party OpenAI-shaped API.

Uses the modern ``openai>=1.0`` client surface (``client.embeddings.create`` /
``client.chat.completions.create``). The SDK is imported lazily so fixerr
installs and runs without it until this provider is actually selected.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..config import load_config
from .base import AIBackend, AIUnavailableError


class OpenAIBackend(AIBackend):
    def __init__(self, compatible: bool = False) -> None:
        section = "openai-compatible" if compatible else "openai"
        cfg: Dict[str, Any] = load_config().get("ai", {}).get(section, {})
        self._compatible = compatible
        self._api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        # Compatible servers point at a custom base_url; plain OpenAI may also
        # set one (e.g. a proxy), so honour it either way when present.
        self._base_url = cfg.get("base_url")
        self._embed_model = cfg.get("embed_model", "text-embedding-3-small")
        self._gen_model = cfg.get("gen_model", "gpt-4o-mini")

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise AIUnavailableError(
                "openai package not installed. Run: pip install fixerr[openai]"
            ) from exc

        # A literal "none" means the compatible server needs no auth; pass a
        # dummy so the client constructs (it validates that a key is present).
        if (self._api_key_env or "").strip().lower() == "none":
            api_key = "not-needed"
        else:
            api_key = os.environ.get(self._api_key_env or "")
            if not api_key:
                raise AIUnavailableError(
                    f"{self._api_key_env} is not set. "
                    f"Export it or set [ai.{'openai-compatible' if self._compatible else 'openai'}] "
                    f"api_key_env in ~/.fixerr/config.toml."
                )

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return openai.OpenAI(**kwargs)

    def embed(self, text: str) -> List[float]:
        client = self._client()
        try:
            resp = client.embeddings.create(model=self._embed_model, input=text)
            return [float(x) for x in resp.data[0].embedding]
        except Exception as exc:
            raise AIUnavailableError(
                f"OpenAI embeddings failed ({exc.__class__.__name__}): {exc}"
            ) from exc

    def generate(self, prompt: str) -> str:
        client = self._client()
        try:
            resp = client.chat.completions.create(
                model=self._gen_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            raise AIUnavailableError(
                f"OpenAI generation failed ({exc.__class__.__name__}): {exc}"
            ) from exc
