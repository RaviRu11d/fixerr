"""``ErnestClient`` — the small, stable API other tools embed.

This is the surface devnest (and any host) depends on: construct a client, then
call :meth:`surface` on a failure's stderr, or :meth:`search` to browse a
project's past errors. Both are defensive — they never raise into the host
command, and they degrade gracefully when the AI backend is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .store import DEFAULT_THRESHOLD, Match, Store


class ErnestClient:
    def __init__(self, db_path: Optional[Path] = None, threshold: float = DEFAULT_THRESHOLD) -> None:
        self._store = Store(db_path)
        self._threshold = threshold

    def surface(self, error_text: str) -> Optional[str]:
        """Return a past fix if ``error_text`` resembles a resolved error, else None.

        Designed to be dropped into a host command's failure path: it never
        raises, and returns ``None`` (rather than erroring) whenever there is no
        confident, resolved match — including when the AI backend is down.
        """
        try:
            if not error_text or not error_text.strip():
                return None
            matches = self._store.search(error_text, top_k=1, resolved_only=True)
            if not matches:
                return None
            top = matches[0]
            if top.score < self._threshold or not top.fix:
                return None
            return top.fix
        except Exception:  # noqa: BLE001 - surfacing must never break the host
            return None

    def search(self, query: str, top_k: int = 3, cwd: Optional[str] = None) -> List[Match]:
        """Return up to ``top_k`` past errors most similar to ``query``.

        Pass ``cwd`` (a project directory) to scope results to errors that
        occurred under that path — used by devnest's per-project drill-down.
        Never raises; returns ``[]`` on any failure.
        """
        try:
            return self._store.search(query, top_k=top_k, cwd=cwd)
        except Exception:  # noqa: BLE001
            return []

    def list_errors(self, cwd: Optional[str] = None) -> List[Match]:
        """All errors (optionally scoped to ``cwd``), most recent first. Never raises."""
        try:
            return self._store.list_errors(cwd=cwd)
        except Exception:  # noqa: BLE001
            return []

    # -- write helpers (used by the CLI / dashboard) ------------------------
    def record(
        self,
        failing_command: str,
        error_text: str,
        cwd: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> int:
        return self._store.add_error(failing_command, error_text, cwd=cwd, exit_code=exit_code)

    def resolve(self, error_id: int, fix_text: str) -> bool:
        return self._store.set_fix(error_id, fix_text)

    def dismiss(self, error_id: int) -> bool:
        return self._store.dismiss(error_id)

    def reopen(self, error_id: int) -> bool:
        return self._store.reopen(error_id)
