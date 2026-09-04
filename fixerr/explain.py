"""AI-generated explanations for a captured error.

Shared by the CLI (``fixerr explain``, auto-triggered from ``fixerr run``)
and the dashboard (the 'e' keybinding) — one place that builds the prompt and
calls the configured generation backend, so both surfaces produce the same
explanation for the same error.
"""

from __future__ import annotations

import re
from typing import Optional

from .ai import get_backend

# Reasoning models (e.g. deepseek-r1) emit <think>...</think> scratch-work
# before the real answer — strip it so the stored explanation is just the
# explanation, not the model's internal monologue.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_PROMPT_TEMPLATE = """You are helping a developer understand a command-line error.

Command: {command}

Error output:
{error}
{fix_section}
In 2-3 sentences, explain what most likely caused this error. If a fix is \
given above, briefly explain why it worked. Be concise and concrete — no \
preamble, no restating the error text back verbatim."""


def build_prompt(command: Optional[str], error_text: str, fix_text: Optional[str] = None) -> str:
    fix_section = f"\nHow it was fixed:\n{fix_text}\n" if fix_text else ""
    return _PROMPT_TEMPLATE.format(
        command=command or "(unknown)",
        error=(error_text or "").strip() or "(no error text captured)",
        fix_section=fix_section,
    )


def generate_explanation_verbose(
    command: Optional[str], error_text: str, fix_text: Optional[str] = None
) -> "tuple[Optional[str], Optional[str]]":
    """Like :func:`generate_explanation`, but also returns *why* it failed.

    Returns ``(explanation, error_detail)`` — exactly one is non-None (unless
    ``error_text`` is empty, in which case both are None). Callers that want
    to surface a real diagnostic (e.g. the dashboard's notification) use this
    instead of the silent wrapper below.
    """
    if not (error_text or "").strip():
        return None, None
    try:
        prompt = build_prompt(command, error_text, fix_text)
        text = get_backend().generate(prompt)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not raised
        return None, f"{exc.__class__.__name__}: {exc}"
    text = _THINK_RE.sub("", text or "").strip()
    if not text:
        return None, "model returned an empty response"
    return text, None


def generate_explanation(
    command: Optional[str], error_text: str, fix_text: Optional[str] = None
) -> Optional[str]:
    """Return an AI-generated explanation, or None if no backend is reachable.

    Never raises — every caller (CLI, dashboard) treats a missing explanation
    as a normal, displayable state, not an error. Use
    :func:`generate_explanation_verbose` instead if you want to know *why*
    it came back None.
    """
    explanation, _detail = generate_explanation_verbose(command, error_text, fix_text)
    return explanation
