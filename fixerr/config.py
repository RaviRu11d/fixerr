"""fixerr configuration — ``~/.fixerr/config.toml``.

Read-only access goes through :func:`load_config`; single-value writes go
through :func:`set_value` (used by ``fixerr config set``). The file is optional:
when it is absent, :data:`DEFAULTS` is returned so a fresh install works with no
setup (defaulting to a local Ollama backend).

Only the standard library is used for *reading* on 3.11+ (``tomllib``); the
``tomli`` backport is used on older interpreters. Writing uses a small,
self-contained serializer (:func:`_dumps`) so no write-capable TOML dependency
is required.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    import tomli as _toml  # type: ignore[no-redef]


# Sensible zero-config defaults: a local Ollama install, no API keys required.
DEFAULTS: Dict[str, Any] = {
    "ai": {
        "provider": "ollama",
        # Used when the active provider cannot embed (notably "anthropic").
        "embed_fallback": "ollama",
        "ollama": {
            "host": "http://localhost:11434",
            "embed_model": "nomic-embed-text",
            "gen_model": "llama3",
            "embed_timeout": 30,
            # Generous default: generation callers (`explain`, `run`) already
            # run detached, so a slow local model isn't a UX problem.
            "gen_timeout": 180,
        },
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "embed_model": "text-embedding-3-small",
            "gen_model": "gpt-4o-mini",
        },
        "openai-compatible": {
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "http://localhost:8000/v1",
            "embed_model": "text-embedding-3-small",
            "gen_model": "gpt-4o-mini",
        },
        "anthropic": {
            "api_key_env": "ANTHROPIC_API_KEY",
            "gen_model": "claude-haiku-4-5",
        },
        "gemini": {
            "api_key_env": "GEMINI_API_KEY",
            "embed_model": "models/text-embedding-004",
            "gen_model": "gemini-1.5-flash",
        },
    },
    # Tier 1 auto-capture policy (the `shell-init` hook + `_auto_capture`).
    # Scoped to Tier 1 only — `fixerr run` (Tier 2) is an explicit, deliberate
    # invocation and is never filtered by these.
    "capture": {
        "auto_capture": True,
        "min_exit_code": 1,
        "ignore_commands": [
            "cd", "ls", "cat", "echo", "man", "git log",
            "git status", "git diff", "clear", "exit",
        ],
        "surface_threshold": 1,
        "quiet": False,
        "ignore_patterns": {
            "patterns": ["^vim ", "^nano ", "^less "],
        },
    },
}


def config_path() -> Path:
    override = os.environ.get("fixerr_CONFIG")
    return Path(override) if override else Path.home() / ".fixerr" / "config.toml"


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config() -> Dict[str, Any]:
    """Return the effective config: :data:`DEFAULTS` deep-merged with the file.

    Missing or malformed files degrade to defaults rather than raising, so a
    broken config never takes down the host command.
    """
    path = config_path()
    if not path.is_file():
        return copy.deepcopy(DEFAULTS)
    try:
        with path.open("rb") as fh:
            loaded = _toml.load(fh)
    except (OSError, ValueError, _toml.TOMLDecodeError):
        return copy.deepcopy(DEFAULTS)
    return _deep_merge(DEFAULTS, loaded)


# ------------------------------------------------------------------ writing ---

def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dumps(data: Dict[str, Any]) -> str:
    """Serialize a dict-of-dicts-of-scalars into TOML with nested tables."""
    lines: list[str] = []

    def dump_table(prefix: str, table: Dict[str, Any]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        subtables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_fmt(value)}")
        if prefix or scalars:
            lines.append("")
        for key, value in subtables.items():
            dump_table(f"{prefix}.{key}" if prefix else key, value)

    dump_table("", data)
    return "\n".join(lines).rstrip("\n") + "\n"


def _read_raw() -> Dict[str, Any]:
    """Read the file as-is (no defaults merged), for round-tripping writes."""
    path = config_path()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return _toml.load(fh)
    except (OSError, ValueError, _toml.TOMLDecodeError):
        return {}


def _coerce_scalar(value: str) -> Any:
    """Best-effort coerce a CLI-supplied string to a TOML-native type.

    Without this, ``config set capture.auto_capture false`` would store the
    literal *string* ``"false"`` — truthy in Python — so every boolean/int
    config key would silently fail to gate anything downstream. bool is
    checked before int/float since ``True``/``False`` would otherwise never
    be reached (str values here, so no bool-is-a-subclass-of-int ambiguity).
    """
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def set_value(dotted_key: str, value: str) -> None:
    """Set a single dotted key (e.g. ``ai.provider``) and persist to disk.

    Only the on-disk keys are round-tripped; unspecified keys keep falling
    back to :data:`DEFAULTS` at read time, so the file stays minimal.
    """
    parts = dotted_key.split(".")
    if not parts or any(not p for p in parts):
        raise ValueError(f"Invalid config key: {dotted_key!r}")

    data = _read_raw()
    cursor = data
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = _coerce_scalar(value)

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(data), encoding="utf-8")


# ------------------------------------------------------------------ masking ---

_SECRET_HINTS = ("key", "token", "secret", "password")


def masked_config() -> Dict[str, Any]:
    """Return the effective config with any secret-looking *values* masked.

    Note: config stores the *env var name* holding the key (``api_key_env``),
    not the key itself, so little is secret here — but we still mask any value
    that itself looks like a credential, and we redact the resolved env values.
    """
    cfg = copy.deepcopy(load_config())

    def scrub(table: Dict[str, Any]) -> None:
        for key, value in list(table.items()):
            if isinstance(value, dict):
                scrub(value)
            elif key == "api_key_env" and isinstance(value, str):
                # Show the env var name, plus whether it is currently set.
                resolved = os.environ.get(value)
                table[key] = f"{value} ({'set' if resolved else 'unset'})"
            elif any(h in key.lower() for h in _SECRET_HINTS) and value:
                table[key] = "****"

    scrub(cfg)
    return cfg
