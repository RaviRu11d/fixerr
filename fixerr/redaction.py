"""Redaction + normalization for error text.

``redact`` strips likely secrets before anything is stored (fixerr keeps a
local error/fix history, and error output routinely contains tokens, keys, and
absolute home paths). ``normalize`` additionally collapses volatile bits
(numbers, hex, paths, memory addresses) so that two runs of the *same* error
embed and match to each other despite differing line numbers or temp paths.
"""

from __future__ import annotations

import re

# --- secret-shaped tokens -------------------------------------------------
_PATTERNS = [
    # key = value / key: value where the key looks sensitive
    (re.compile(r"(?i)(\b(?:api[_-]?key|token|secret|password|passwd|authorization|bearer)"
                r"\s*[:=]\s*)(\S+)"), r"\1<redacted>"),
    # Standalone high-entropy provider keys
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "<redacted>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<redacted>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted>"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"), "<redacted>"),
    # Emails
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<email>"),
]


def redact(text: str) -> str:
    """Remove likely secrets/PII from ``text`` for safe local storage."""
    if not text:
        return ""
    out = text
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    return out


# --- normalization --------------------------------------------------------
_HOME = re.compile(r"/(?:home|Users)/[^/\s]+")
_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_NUM = re.compile(r"\b\d+\b")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case ``text`` and collapse volatile tokens to stable placeholders."""
    if not text:
        return ""
    out = redact(text)
    out = _HOME.sub("<home>", out)
    out = _HEX.sub("<hex>", out)
    out = _NUM.sub("<n>", out)
    out = _WS.sub(" ", out)
    return out.strip().lower()
