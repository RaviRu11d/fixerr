"""SQLite-backed error/fix store with semantic + text fallback search.

Errors and their fixes live in ``~/.fixerr/errors.db``. Rows carry an optional
embedding (via the configured embed backend). Search prefers cosine similarity
over embeddings, but when the AI backend is unavailable (e.g. Ollama down and no
fallback) it degrades to token-overlap text matching so history stays usable
offline.

Schema notes: a row's lifecycle is tracked by ``status`` (``open`` ->
``resolved`` or ``wont-fix``), not a boolean — ``Match.resolved`` is a computed
convenience property over it. Columns are added additively via
:func:`_migrate` so existing databases upgrade in place without data loss.
"""

from __future__ import annotations

import math
import os
import sqlite3
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .ai import AIUnavailableError, get_embed_backend
from .fingerprint import compute_fingerprint
from .redaction import normalize, redact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cwd TEXT,
    failing_command TEXT,
    error_text_redacted TEXT,
    error_text_normalized TEXT,
    embedding BLOB,
    fix_text TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    resolved_at TEXT,
    exit_code INTEGER,
    git_commit TEXT,
    ai_explanation TEXT,
    fingerprint TEXT,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen TEXT,
    last_seen TEXT
);
"""

# Columns added after the original release, applied idempotently to existing
# databases via ALTER TABLE (SQLite has no CREATE-OR-ALTER, so new installs get
# them from _SCHEMA above and old ones get them patched in here).
_ADDED_COLUMNS = {
    "status": "TEXT NOT NULL DEFAULT 'open'",
    "resolved_at": "TEXT",
    "exit_code": "INTEGER",
    "git_commit": "TEXT",
    "ai_explanation": "TEXT",
    "fingerprint": "TEXT",
    "occurrence_count": "INTEGER NOT NULL DEFAULT 1",
    "first_seen": "TEXT",
    "last_seen": "TEXT",
}

STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"
STATUS_WONT_FIX = "wont-fix"

DEFAULT_THRESHOLD = float(os.environ.get("fixerr_SIM_THRESHOLD", "0.7"))


@dataclass
class Match:
    """A search hit / list row, shaped for display."""

    id: int
    timestamp: str
    cwd: Optional[str]
    failing_command: Optional[str]
    error: str
    fix: str
    status: str
    score: float
    exit_code: Optional[int] = None
    git_commit: Optional[str] = None
    resolved_at: Optional[str] = None
    ai_explanation: Optional[str] = None
    fingerprint: Optional[str] = None
    occurrence_count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.status == STATUS_RESOLVED

    @property
    def date(self) -> str:
        ts = self.last_seen or self.timestamp
        try:
            return datetime.fromisoformat(ts).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return (ts or "")[:10]


def db_path() -> Path:
    override = os.environ.get("fixerr_DB")
    return Path(override) if override else Path.home() / ".fixerr" / "errors.db"


def _pack(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: Optional[bytes]) -> List[float]:
    if not blob:
        return []
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _token_overlap(query: str, candidate: str) -> float:
    """Jaccard overlap of normalized tokens — the offline fallback ranking."""
    qs = set(normalize(query).split())
    cs = set(normalize(candidate).split())
    if not qs or not cs:
        return 0.0
    return len(qs & cs) / len(qs | cs)


def _git_commit(cwd: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class Store:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else db_path()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(_SCHEMA)
        self._migrate(conn)
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(errors)").fetchall()}
        # Pre-dashboard databases used a `resolved` boolean instead of `status`;
        # backfill it once, right after the column is added below.
        had_legacy_resolved = "resolved" in cols and "status" not in cols
        for col, ddl in _ADDED_COLUMNS.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE errors ADD COLUMN {col} {ddl}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_errors_fingerprint ON errors(fingerprint)")
        if had_legacy_resolved:
            conn.execute(
                f"UPDATE errors SET status = '{STATUS_RESOLVED}' "
                f"WHERE resolved = 1 AND status = '{STATUS_OPEN}'"
            )
        # Backfill occurrence and timing fields on legacy rows
        conn.execute("UPDATE errors SET first_seen = timestamp WHERE first_seen IS NULL")
        conn.execute("UPDATE errors SET last_seen = timestamp WHERE last_seen IS NULL")
        conn.execute("UPDATE errors SET occurrence_count = 1 WHERE occurrence_count IS NULL")

        # Backfill fingerprints for legacy rows
        unfingerprinted = conn.execute(
            "SELECT id, failing_command, error_text_redacted, exit_code FROM errors WHERE fingerprint IS NULL"
        ).fetchall()
        for row in unfingerprinted:
            fp = compute_fingerprint(
                row["failing_command"],
                row["error_text_redacted"],
                row["exit_code"],
            )
            conn.execute("UPDATE errors SET fingerprint = ? WHERE id = ?", (fp, row["id"]))

    # -- writes ------------------------------------------------------------
    def _try_embed(self, text: str) -> Optional[List[float]]:
        try:
            return get_embed_backend().embed(normalize(text))
        except (AIUnavailableError, NotImplementedError, Exception):  # noqa: BLE001
            return None

    def add_error(
        self,
        failing_command: str,
        error_text: str,
        cwd: Optional[str] = None,
        exit_code: Optional[int] = None,
        git_commit: Optional[str] = None,
    ) -> int:
        """Full capture path: redacts, normalizes, embeds, and deduplicates ``error_text``.

        If an error with the same fingerprint already exists, increments its
        ``occurrence_count`` and updates ``last_seen`` rather than storing a
        duplicate row. If the existing row was a Tier 1 capture (missing
        redacted error text / embedding), it is upgraded in place.
        """
        cwd = cwd or os.getcwd()
        resolved_git_commit = git_commit if git_commit is not None else _git_commit(cwd)
        fp = compute_fingerprint(failing_command, error_text, exit_code)
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, error_text_redacted, embedding FROM errors WHERE fingerprint = ? ORDER BY id DESC LIMIT 1",
                (fp,),
            ).fetchone()

            if existing is not None:
                existing_id = int(existing["id"])
                # If existing row didn't have error text / embedding, upgrade it now
                if not existing["error_text_redacted"] and error_text:
                    embedding = self._try_embed(error_text)
                    conn.execute(
                        """UPDATE errors SET
                           occurrence_count = occurrence_count + 1,
                           last_seen = ?,
                           cwd = ?,
                           exit_code = COALESCE(?, exit_code),
                           git_commit = COALESCE(?, git_commit),
                           error_text_redacted = ?,
                           error_text_normalized = ?,
                           embedding = ?
                           WHERE id = ?""",
                        (
                            now,
                            cwd,
                            exit_code,
                            resolved_git_commit,
                            redact(error_text),
                            normalize(error_text),
                            _pack(embedding) if embedding else None,
                            existing_id,
                        ),
                    )
                else:
                    conn.execute(
                        """UPDATE errors SET
                           occurrence_count = occurrence_count + 1,
                           last_seen = ?,
                           cwd = ?,
                           exit_code = COALESCE(?, exit_code),
                           git_commit = COALESCE(?, git_commit)
                           WHERE id = ?""",
                        (now, cwd, exit_code, resolved_git_commit, existing_id),
                    )
                return existing_id

            embedding = self._try_embed(error_text)
            cur = conn.execute(
                """INSERT INTO errors
                   (timestamp, cwd, failing_command, error_text_redacted,
                    error_text_normalized, embedding, fix_text, status,
                    exit_code, git_commit, fingerprint, occurrence_count,
                    first_seen, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, 1, ?, ?)""",
                (
                    now,
                    cwd,
                    failing_command,
                    redact(error_text),
                    normalize(error_text),
                    _pack(embedding) if embedding else None,
                    STATUS_OPEN,
                    exit_code,
                    resolved_git_commit,
                    fp,
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def auto_capture_tier1(
        self,
        failing_command: str,
        exit_code: int,
        cwd: str,
        git_commit: Optional[str] = None,
        dedup_window_seconds: int = 5,
        match_limit: int = 3,
    ) -> Tuple[bool, List[Match]]:
        """Fast, SQL-only auto-capture fired on every failed shell command.

        Returns ``(inserted, matches)``:
          - ``inserted=False`` when this exact command + exit_code was already
            captured within ``dedup_window_seconds`` — nothing is written,
            ``matches`` is empty.
          - When captured outside ``dedup_window_seconds``: if this exact error
            has been seen before, increments ``occurrence_count`` and updates
            ``last_seen``; otherwise creates a new row.
          - ``matches`` are past *resolved* errors with the same
            ``failing_command`` (exact string match only).
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        cutoff = (now - timedelta(seconds=dedup_window_seconds)).isoformat()
        fp = compute_fingerprint(failing_command, None, exit_code)

        with self._connect() as conn:
            # 1. Dedup within immediate window (anti-spam)
            dup = conn.execute(
                """SELECT 1 FROM errors
                   WHERE (fingerprint = ? OR (failing_command = ? AND exit_code = ? AND cwd = ?))
                     AND last_seen >= ? LIMIT 1""",
                (fp, failing_command, exit_code, cwd, cutoff),
            ).fetchone()
            if dup is not None:
                return False, []

            # 2. Check if existing row with this fingerprint exists
            existing = conn.execute(
                "SELECT id FROM errors WHERE fingerprint = ? ORDER BY id DESC LIMIT 1",
                (fp,),
            ).fetchone()

            if existing is not None:
                conn.execute(
                    """UPDATE errors SET
                       occurrence_count = occurrence_count + 1,
                       last_seen = ?,
                       cwd = ?,
                       git_commit = COALESCE(?, git_commit)
                       WHERE id = ?""",
                    (now_iso, cwd, git_commit or None, int(existing["id"])),
                )
            else:
                conn.execute(
                    """INSERT INTO errors
                       (timestamp, cwd, failing_command, error_text_redacted,
                        error_text_normalized, embedding, fix_text, status,
                        exit_code, git_commit, fingerprint, occurrence_count,
                        first_seen, last_seen)
                       VALUES (?, ?, ?, NULL, NULL, NULL, '', ?, ?, ?, ?, 1, ?, ?)""",
                    (now_iso, cwd, failing_command, STATUS_OPEN, exit_code, git_commit or None, fp, now_iso, now_iso),
                )

            rows = conn.execute(
                """SELECT * FROM errors
                   WHERE failing_command = ? AND status = ?
                   ORDER BY resolved_at DESC LIMIT ?""",
                (failing_command, STATUS_RESOLVED, match_limit),
            ).fetchall()
        return True, [self._to_match(row, score=0.0) for row in rows]

    def set_fix(self, error_id: int, fix_text: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE errors SET fix_text = ?, status = ?, resolved_at = ? WHERE id = ?",
                (fix_text, STATUS_RESOLVED, datetime.now(timezone.utc).isoformat(), error_id),
            )
            return cur.rowcount > 0

    def dismiss(self, error_id: int) -> bool:
        """Mark an error as won't-fix (won't be surfaced or resolved)."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE errors SET status = ? WHERE id = ?", (STATUS_WONT_FIX, error_id)
            )
            return cur.rowcount > 0

    def reopen(self, error_id: int) -> bool:
        """Revert a resolved or won't-fix error back to open.

        ``fix_text`` is kept (not cleared) — reopening usually means the fix
        didn't fully hold, and the old note is still useful context; it gets
        overwritten if/when the error is resolved again via `set_fix`.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE errors SET status = ?, resolved_at = NULL WHERE id = ?",
                (STATUS_OPEN, error_id),
            )
            return cur.rowcount > 0

    def set_ai_explanation(self, error_id: int, explanation: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE errors SET ai_explanation = ? WHERE id = ?", (explanation, error_id)
            )
            return cur.rowcount > 0

    def get(self, error_id: int) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM errors WHERE id = ?", (error_id,)).fetchone()

    # -- reads ---------------------------------------------------------------
    def _rows(self, cwd: Optional[str] = None) -> List[sqlite3.Row]:
        with self._connect() as conn:
            if cwd:
                # Scope to errors under this project's directory (prefix match).
                like = cwd.rstrip("/") + "%"
                return conn.execute(
                    "SELECT * FROM errors WHERE cwd LIKE ?", (like,)
                ).fetchall()
            return conn.execute("SELECT * FROM errors").fetchall()

    def list_errors(self, cwd: Optional[str] = None) -> List[Match]:
        """All errors (optionally scoped to ``cwd``), most recently active first."""
        rows = sorted(
            self._rows(cwd),
            key=lambda r: (r["last_seen"] if "last_seen" in r.keys() and r["last_seen"] else r["timestamp"]),
            reverse=True,
        )
        return [self._to_match(row, score=0.0) for row in rows]

    def list_with_embeddings(self, cwd: Optional[str] = None) -> List[tuple]:
        """(Match, vector) pairs for every error that has a stored embedding.

        Used for clustering — rows captured while the embed backend was down
        have no vector and are silently excluded.
        """
        pairs = []
        for row in self._rows(cwd):
            if row["embedding"]:
                pairs.append((self._to_match(row, score=0.0), _unpack(row["embedding"])))
        return pairs

    def similar_to(self, error_id: int, top_k: int = 3, cwd: Optional[str] = None) -> List[Match]:
        """Errors most similar to ``error_id``, by its already-stored embedding.

        Compares stored vectors directly — no fresh embed call, so this needs
        no live AI backend beyond whatever produced the embeddings originally.
        Returns ``[]`` when ``error_id`` has no embedding (never embedded, or
        embedding failed at capture time).
        """
        with self._connect() as conn:
            target = conn.execute("SELECT * FROM errors WHERE id = ?", (error_id,)).fetchone()
        if target is None or not target["embedding"]:
            return []
        target_vec = _unpack(target["embedding"])

        scored: List[Match] = []
        for row in self._rows(cwd):
            if int(row["id"]) == error_id or not row["embedding"]:
                continue
            score = _cosine(target_vec, _unpack(row["embedding"]))
            scored.append(self._to_match(row, score))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 3,
        cwd: Optional[str] = None,
        resolved_only: bool = False,
    ) -> List[Match]:
        rows = self._rows(cwd)
        if resolved_only:
            rows = [r for r in rows if r["status"] == STATUS_RESOLVED]
        if not rows:
            return []

        # Prefer semantic search; fall back to token overlap when embeddings
        # are unavailable (offline / no embed backend).
        query_vec: Optional[List[float]] = None
        try:
            query_vec = get_embed_backend().embed(normalize(query))
        except (AIUnavailableError, NotImplementedError, Exception):  # noqa: BLE001
            query_vec = None

        scored: List[Match] = []
        for row in rows:
            if query_vec is not None and row["embedding"]:
                score = _cosine(query_vec, _unpack(row["embedding"]))
            else:
                score = _token_overlap(query, row["error_text_normalized"] or "")
            scored.append(self._to_match(row, score))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _to_match(row: sqlite3.Row, score: float) -> Match:
        cols = row.keys() if hasattr(row, "keys") else []
        return Match(
            id=int(row["id"]),
            timestamp=row["timestamp"],
            cwd=row["cwd"],
            failing_command=row["failing_command"],
            error=(row["error_text_redacted"] or "").strip(),
            fix=(row["fix_text"] or "").strip(),
            status=row["status"] or STATUS_OPEN,
            score=float(score),
            exit_code=row["exit_code"] if "exit_code" in cols else None,
            git_commit=row["git_commit"] if "git_commit" in cols else None,
            resolved_at=row["resolved_at"] if "resolved_at" in cols else None,
            ai_explanation=row["ai_explanation"] if "ai_explanation" in cols else None,
            fingerprint=row["fingerprint"] if "fingerprint" in cols else None,
            occurrence_count=int(row["occurrence_count"]) if "occurrence_count" in cols and row["occurrence_count"] is not None else 1,
            first_seen=row["first_seen"] if "first_seen" in cols else row["timestamp"],
            last_seen=row["last_seen"] if "last_seen" in cols else row["timestamp"],
        )
