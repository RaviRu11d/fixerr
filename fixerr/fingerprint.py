"""Deterministic fingerprinting for errors.

A fingerprint is a compact hash that identifies the root cause/signature of an
error independently of volatile elements (line numbers, timestamps, memory
addresses, temp file paths). It enables deduplication and recurrence tracking
across repeated occurrences.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from .redaction import normalize


def compute_fingerprint(
    failing_command: Optional[str],
    error_text: Optional[str],
    exit_code: Optional[int] = None,
) -> str:
    """Compute a deterministic 16-character hex fingerprint.

    If ``error_text`` is present, the fingerprint is derived from the normalized
    command and normalized error text (where secrets, paths, hex, and numbers
    have been collapsed by :func:`fixerr.redaction.normalize`).

    If ``error_text`` is missing (e.g. fast Tier 1 shell capture), the
    fingerprint is derived from the normalized command and exit code.
    """
    cmd = (failing_command or "").strip().lower()
    if error_text and error_text.strip():
        norm_err = normalize(error_text)
        payload = f"cmd:{cmd}\nerr:{norm_err}"
    else:
        ec = str(exit_code) if exit_code is not None else ""
        payload = f"cmd:{cmd}\nexit:{ec}\n<no_error_text>"

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]
