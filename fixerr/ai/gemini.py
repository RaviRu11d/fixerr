"""Google Gemini backend (``google-generativeai``).

The SDK is imported lazily so fixerr runs without it until this provider is
selected. Any connection / auth failure degrades to :class:`AIUnavailableError`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..config import load_config
from .base import AIBackend, AIUnavailableError


class GeminiBackend(AIBackend):
    def __init__(self) -> None:
        cfg: Dict[str, Any] = load_config().get("ai", {}).get("gemini", {})
        self._api_key_env = cfg.get("api_key_env", "GEMINI_API_KEY")
        self._embed_model = cfg.get("embed_model", "models/text-embedding-004")
        self._gen_model = cfg.get("gen_model", "gemini-1.5-flash")

    def _genai(self):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise AIUnavailableError(
                "google-generativeai package not installed. "
                "Run: pip install fixerr[gemini]"
            ) from exc

        api_key = os.environ.get(self._api_key_env or "")
        if not api_key:
            raise AIUnavailableError(
                f"{self._api_key_env} is not set. "
                f"Export it or set [ai.gemini] api_key_env in ~/.fixerr/config.toml."
            )
        genai.configure(api_key=api_key)
        return genai

    def embed(self, text: str) -> List[float]:
        genai = self._genai()
        try:
            result = genai.embed_content(model=self._embed_model, content=text)
            return [float(x) for x in result["embedding"]]
        except Exception as exc:
            raise AIUnavailableError(
                f"Gemini embeddings failed ({exc.__class__.__name__}): {exc}"
            ) from exc

    def generate(self, prompt: str) -> str:
        genai = self._genai()
        try:
            model = genai.GenerativeModel(self._gen_model)
            return model.generate_content(prompt).text
        except Exception as exc:
            raise AIUnavailableError(
                f"Gemini generation failed ({exc.__class__.__name__}): {exc}"
            ) from exc
