"""One test per AI backend, plus the factory's embed-fallback routing.

Every provider SDK / HTTP call is mocked — no real network or API calls. Each
backend is checked for (a) correct return types from ``embed``/``generate`` and
(b) raising :class:`AIUnavailableError` on a connection failure.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from fixerr.ai import reset_cache
from fixerr.ai.base import AIUnavailableError


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    # No config file -> defaults; isolate from the user's ~/.fixerr.
    monkeypatch.setenv("fixerr_CONFIG", str(tmp_path / "config.toml"))
    reset_cache()
    yield
    reset_cache()


# ------------------------------------------------------------------ ollama ---

def test_ollama_embed_and_generate():
    from fixerr.ai.ollama import OllamaBackend

    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json = mock.Mock(return_value={"embedding": [0.1, 0.2, 0.3], "response": "hello"})

    backend = OllamaBackend()
    with mock.patch("httpx.post", return_value=resp):
        vec = backend.embed("boom")
        text = backend.generate("boom")

    assert isinstance(vec, list) and all(isinstance(x, float) for x in vec)
    assert vec == [0.1, 0.2, 0.3]
    assert isinstance(text, str) and text == "hello"


def test_ollama_unavailable_raises():
    import httpx

    from fixerr.ai.ollama import OllamaBackend

    backend = OllamaBackend()
    with mock.patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(AIUnavailableError):
            backend.embed("boom")
        with pytest.raises(AIUnavailableError):
            backend.generate("boom")


def test_ollama_timeout_raises_with_a_distinct_message():
    """A slow-but-reachable model should be diagnosable as a timeout, not
    reported as the generic "is it running?" unreachable message."""
    import httpx

    from fixerr.ai.ollama import OllamaBackend

    backend = OllamaBackend()
    with mock.patch("httpx.post", side_effect=httpx.ReadTimeout("timed out")):
        with pytest.raises(AIUnavailableError, match="didn't respond within"):
            backend.generate("boom")


def test_ollama_gen_timeout_is_configurable_and_generous_by_default():
    from fixerr.ai.ollama import OllamaBackend

    backend = OllamaBackend()
    assert backend._gen_timeout == 180  # generous default — generate() runs detached
    assert backend._embed_timeout == 30

    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json = mock.Mock(return_value={"response": "hi"})
    with mock.patch("httpx.post", return_value=resp) as post:
        backend.generate("boom")
        _, kwargs = post.call_args
        assert kwargs["timeout"] == 180


# ------------------------------------------------------------------ openai ---

def _fake_openai():
    mod = types.ModuleType("openai")
    client = mock.Mock()
    client.embeddings.create.return_value = mock.Mock(data=[mock.Mock(embedding=[0.4, 0.5])])
    client.chat.completions.create.return_value = mock.Mock(
        choices=[mock.Mock(message=mock.Mock(content="generated"))]
    )
    mod.OpenAI = mock.Mock(return_value=client)
    return mod, client


def test_openai_embed_and_generate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mod, client = _fake_openai()
    monkeypatch.setitem(sys.modules, "openai", mod)

    from fixerr.ai.openai import OpenAIBackend

    backend = OpenAIBackend()
    vec = backend.embed("boom")
    text = backend.generate("boom")

    assert vec == [0.4, 0.5] and all(isinstance(x, float) for x in vec)
    assert text == "generated"


def test_openai_unavailable_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mod, client = _fake_openai()
    client.embeddings.create.side_effect = RuntimeError("connection reset")
    monkeypatch.setitem(sys.modules, "openai", mod)

    from fixerr.ai.openai import OpenAIBackend

    with pytest.raises(AIUnavailableError):
        OpenAIBackend().embed("boom")


def test_openai_compatible_dummy_key(monkeypatch):
    # api_key_env == "none" -> a dummy key is passed, no env var required.
    monkeypatch.setenv("fixerr_CONFIG", "")  # trigger defaults path below
    mod, client = _fake_openai()
    monkeypatch.setitem(sys.modules, "openai", mod)

    from fixerr.ai.openai import OpenAIBackend

    backend = OpenAIBackend(compatible=True)
    backend._api_key_env = "none"  # simulate config value
    backend.embed("boom")
    # Constructed with a dummy key, not a real env var.
    _, kwargs = mod.OpenAI.call_args
    assert kwargs["api_key"] == "not-needed"


# --------------------------------------------------------------- anthropic ---

def _fake_anthropic():
    mod = types.ModuleType("anthropic")
    client = mock.Mock()
    client.messages.create.return_value = mock.Mock(content=[mock.Mock(text="claude says hi")])
    mod.Anthropic = mock.Mock(return_value=client)
    return mod, client


def test_anthropic_generate_and_embed_notimplemented(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
    mod, client = _fake_anthropic()
    monkeypatch.setitem(sys.modules, "anthropic", mod)

    from fixerr.ai.anthropic import AnthropicBackend

    backend = AnthropicBackend()
    assert backend.generate("boom") == "claude says hi"
    # max_tokens is pinned to 512 by contract.
    _, kwargs = client.messages.create.call_args
    assert kwargs["max_tokens"] == 512

    with pytest.raises(NotImplementedError):
        backend.embed("boom")


def test_anthropic_unavailable_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
    mod, client = _fake_anthropic()
    client.messages.create.side_effect = RuntimeError("network down")
    monkeypatch.setitem(sys.modules, "anthropic", mod)

    from fixerr.ai.anthropic import AnthropicBackend

    with pytest.raises(AIUnavailableError):
        AnthropicBackend().generate("boom")


# ------------------------------------------------------------------ gemini ---

def _install_fake_gemini(monkeypatch):
    genai = types.ModuleType("google.generativeai")
    genai.configure = mock.Mock()
    genai.embed_content = mock.Mock(return_value={"embedding": [0.6, 0.7]})
    model = mock.Mock()
    model.generate_content.return_value = mock.Mock(text="gemini says hi")
    genai.GenerativeModel = mock.Mock(return_value=model)

    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai)
    return genai


def test_gemini_embed_and_generate(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    _install_fake_gemini(monkeypatch)

    from fixerr.ai.gemini import GeminiBackend

    backend = GeminiBackend()
    vec = backend.embed("boom")
    text = backend.generate("boom")

    assert vec == [0.6, 0.7] and all(isinstance(x, float) for x in vec)
    assert text == "gemini says hi"


def test_gemini_unavailable_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    genai = _install_fake_gemini(monkeypatch)
    genai.embed_content.side_effect = RuntimeError("connection failed")

    from fixerr.ai.gemini import GeminiBackend

    with pytest.raises(AIUnavailableError):
        GeminiBackend().embed("boom")


# ----------------------------------------------------------------- factory ---

def test_factory_anthropic_uses_embed_fallback(monkeypatch, tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[ai]\nprovider = "anthropic"\nembed_fallback = "ollama"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("fixerr_CONFIG", str(cfg))
    reset_cache()

    from fixerr.ai.anthropic import AnthropicBackend
    from fixerr.ai.factory import get_backend, get_embed_backend
    from fixerr.ai.ollama import OllamaBackend

    assert isinstance(get_backend(), AnthropicBackend)
    assert isinstance(get_embed_backend(), OllamaBackend)
