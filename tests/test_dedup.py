"""Tests for Phase 1: Error Deduplication & Fingerprinting."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
from typer.testing import CliRunner

from fixerr.cli import app
from fixerr.fingerprint import compute_fingerprint
from fixerr.store import STATUS_RESOLVED, Store

runner = CliRunner()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("fixerr_DB", str(tmp_path / "errors.db"))
    monkeypatch.setenv("fixerr_CONFIG", str(tmp_path / "config.toml"))

    def fake_embed(text: str):
        h = sum(text.encode())
        return [(h % (i + 7)) / 10.0 for i in range(8)]

    fake_backend = mock.Mock()
    fake_backend.embed.side_effect = fake_embed
    monkeypatch.setattr("fixerr.store.get_embed_backend", lambda: fake_backend)
    return Store()


# -------------------------------------------------------- fingerprint tests ---


def test_fingerprint_deterministic():
    fp1 = compute_fingerprint("pytest", "AssertionError: 1 != 2 in test_foo.py:42", exit_code=1)
    fp2 = compute_fingerprint("pytest", "AssertionError: 1 != 2 in test_foo.py:42", exit_code=1)
    assert fp1 == fp2
    assert len(fp1) == 16


def test_fingerprint_collapses_volatile_tokens():
    # Different line numbers, memory addresses, and home paths collapse to same fingerprint
    fp1 = compute_fingerprint("python main.py", "Error at 0x7ffd1234 in /home/alice/app.py line 42")
    fp2 = compute_fingerprint("python main.py", "Error at 0x7ffd9876 in /home/bob/app.py line 99")
    assert fp1 == fp2


def test_fingerprint_differs_for_different_errors():
    fp1 = compute_fingerprint("python main.py", "ImportError: No module named requests")
    fp2 = compute_fingerprint("python main.py", "ConnectionRefusedError: Connection refused")
    assert fp1 != fp2


def test_fingerprint_no_error_text_uses_command_and_exit_code():
    fp1 = compute_fingerprint("docker compose up", None, exit_code=1)
    fp2 = compute_fingerprint("docker compose up", None, exit_code=1)
    fp3 = compute_fingerprint("docker compose up", None, exit_code=2)
    assert fp1 == fp2
    assert fp1 != fp3


# ----------------------------------------------------- store add_error dedup ---


def test_add_error_first_time_creates_new_row(store):
    eid = store.add_error("npm run build", "SyntaxError: Unexpected token", exit_code=1)
    row = store.get(eid)
    assert row is not None
    assert row["occurrence_count"] == 1
    assert row["first_seen"] is not None
    assert row["last_seen"] is not None
    assert row["fingerprint"] is not None


def test_add_error_duplicate_increments_occurrence(store):
    eid1 = store.add_error("npm run build", "SyntaxError: Unexpected token", exit_code=1)
    eid2 = store.add_error("npm run build", "SyntaxError: Unexpected token", exit_code=1)
    assert eid1 == eid2

    row = store.get(eid1)
    assert row["occurrence_count"] == 2
    # Only 1 row in the store
    assert len(store.list_errors()) == 1


def test_add_error_duplicate_updates_last_seen_and_context(store):
    eid1 = store.add_error("git push", "remote rejected: pre-receive hook declined", cwd="/repo1", git_commit="c0ffee")
    eid2 = store.add_error("git push", "remote rejected: pre-receive hook declined", cwd="/repo2", git_commit="deadbeef")
    assert eid1 == eid2

    row = store.get(eid1)
    assert row["occurrence_count"] == 2
    assert row["cwd"] == "/repo2"
    assert row["git_commit"] == "deadbeef"


def test_add_error_different_error_creates_separate_record(store):
    eid1 = store.add_error("python app.py", "KeyError: 'user_id'")
    eid2 = store.add_error("python app.py", "IndexError: list index out of range")
    assert eid1 != eid2
    assert len(store.list_errors()) == 2


def test_add_error_resolved_error_increments_count_without_reopening(store):
    eid1 = store.add_error("docker compose up", "port 5432 already in use")
    store.set_fix(eid1, "kill stale postgres container")

    row = store.get(eid1)
    assert row["status"] == STATUS_RESOLVED

    # Recurrence of resolved error
    eid2 = store.add_error("docker compose up", "port 5432 already in use")
    assert eid1 == eid2
    row2 = store.get(eid1)
    assert row2["occurrence_count"] == 2
    assert row2["status"] == STATUS_RESOLVED
    assert row2["fix_text"] == "kill stale postgres container"


def test_add_error_upgrades_tier1_placeholder(store):
    # Suppose Tier 1 shell capture recorded command only
    inserted, _ = store.auto_capture_tier1("cargo build", 101, "/tmp/rust_app")
    assert inserted is True
    errors = store.list_errors()
    assert len(errors) == 1
    tier1_row = errors[0]
    assert tier1_row.error == ""

    # Now Tier 2 or manual capture runs with full stderr
    eid = store.add_error("cargo build", "error[E0308]: mismatched types", cwd="/tmp/rust_app", exit_code=101)
    row = store.get(eid)
    assert "mismatched types" in (row["error_text_redacted"] or "")
    assert row["occurrence_count"] >= 1


# ---------------------------------------------------- auto_capture_tier1 tests ---


def test_auto_capture_tier1_dedup_and_recurrence(store):
    # First capture
    inserted1, _ = store.auto_capture_tier1("make test", 2, "/tmp/proj", dedup_window_seconds=1)
    assert inserted1 is True
    assert len(store.list_errors()) == 1

    # Immediate duplicate within dedup window is suppressed
    inserted2, _ = store.auto_capture_tier1("make test", 2, "/tmp/proj", dedup_window_seconds=10)
    assert inserted2 is False

    # Simulate passage of time for next capture
    now = datetime.now(timezone.utc)
    with store._connect() as conn:
        conn.execute(
            "UPDATE errors SET last_seen = ? WHERE failing_command = 'make test'",
            ((now - timedelta(seconds=20)).isoformat(),),
        )

    # Capture after window updates occurrence_count instead of creating a second row
    inserted3, _ = store.auto_capture_tier1("make test", 2, "/tmp/proj", dedup_window_seconds=5)
    assert inserted3 is True
    errors = store.list_errors()
    assert len(errors) == 1
    assert errors[0].occurrence_count == 2


# ------------------------------------------------------ schema migration tests ---


def test_legacy_schema_migration(tmp_path):
    db_file = tmp_path / "legacy.db"
    # Create legacy table schema without fingerprint / occurrence columns
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """CREATE TABLE errors (
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
            ai_explanation TEXT
        )"""
    )
    conn.execute(
        """INSERT INTO errors
           (timestamp, cwd, failing_command, error_text_redacted, error_text_normalized, fix_text, status)
           VALUES ('2025-01-01T00:00:00Z', '/tmp', 'pytest', 'AssertionError: test failed', 'assertionerror: test failed', '', 'open')"""
    )
    conn.commit()
    conn.close()

    # Instantiate Store over the legacy database to trigger migration
    store = Store(db_file)
    row = store.get(1)
    assert row["fingerprint"] is not None
    assert row["occurrence_count"] == 1
    assert row["first_seen"] == "2025-01-01T00:00:00Z"
    assert row["last_seen"] == "2025-01-01T00:00:00Z"


def test_legacy_migration_with_duplicate_records(tmp_path):
    db_file = tmp_path / "legacy_dups.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """CREATE TABLE errors (
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
            ai_explanation TEXT
        )"""
    )
    # Insert two historical duplicate rows
    conn.execute(
        """INSERT INTO errors
           (timestamp, cwd, failing_command, error_text_redacted, error_text_normalized, fix_text, status)
           VALUES ('2025-01-01T00:00:00Z', '/tmp', 'npm test', 'Jest failed', 'jest failed', '', 'open')"""
    )
    conn.execute(
        """INSERT INTO errors
           (timestamp, cwd, failing_command, error_text_redacted, error_text_normalized, fix_text, status)
           VALUES ('2025-01-02T00:00:00Z', '/tmp', 'npm test', 'Jest failed', 'jest failed', '', 'open')"""
    )
    conn.commit()
    conn.close()

    store = Store(db_file)
    r1 = store.get(1)
    r2 = store.get(2)
    # Both legacy rows receive the computed fingerprint
    assert r1["fingerprint"] == r2["fingerprint"]
    assert r1["occurrence_count"] == 1
    assert r2["occurrence_count"] == 1

    # Adding the error again targets the latest row (id=2) and increments its count
    eid = store.add_error("npm test", "Jest failed", cwd="/tmp")
    assert eid == 2
    assert store.get(2)["occurrence_count"] == 2
    assert store.get(1)["occurrence_count"] == 1


# ------------------------------------------------- edge-case & collision tests ---


def test_fingerprint_collision_and_edge_cases():
    # 1. Very large error trace (100KB)
    large_error = "Traceback (most recent call last):\n" + ("  File 'app.py', line 1, in foo\n" * 2000) + "ZeroDivisionError: division by zero"
    fp_large1 = compute_fingerprint("python main.py", large_error)
    fp_large2 = compute_fingerprint("python main.py", large_error)
    assert fp_large1 == fp_large2
    assert len(fp_large1) == 16

    # 2. Non-ASCII, Unicode, and Emojis
    unicode_err = "Error: 💥 DB connection failed (エラー: 接続タイムアウト) in café.py"
    fp_u1 = compute_fingerprint("cargo run", unicode_err)
    fp_u2 = compute_fingerprint("cargo run", unicode_err)
    assert fp_u1 == fp_u2
    assert len(fp_u1) == 16

    # 3. Volatile addresses, line numbers, and paths collapse identically
    trace1 = "Error at 0x7fff5fbff8c0 in /home/alice/app.py:100 (exit code 137)"
    trace2 = "Error at 0x7fff99999999 in /Users/bob/app.py:450 (exit code 137)"
    assert compute_fingerprint("run.sh", trace1) == compute_fingerprint("run.sh", trace2)

    # 4. Command casing and extra spaces are normalized
    assert compute_fingerprint("  DOCKER Compose Up  ", "error msg") == compute_fingerprint("docker compose up", "error msg")


def test_same_error_across_different_commands_has_separate_fingerprints_and_records(store):
    err = "FATAL: password authentication failed for user 'postgres'"
    cmd_a = "psql -U postgres -h localhost"
    cmd_b = "python manage.py migrate"

    fp_a = compute_fingerprint(cmd_a, err)
    fp_b = compute_fingerprint(cmd_b, err)
    assert fp_a != fp_b

    eid_a = store.add_error(cmd_a, err)
    eid_b = store.add_error(cmd_b, err)
    assert eid_a != eid_b
    assert len(store.list_errors()) == 2

    # Recurrence of command A only affects command A
    eid_a2 = store.add_error(cmd_a, err)
    assert eid_a2 == eid_a
    assert store.get(eid_a)["occurrence_count"] == 2
    assert store.get(eid_b)["occurrence_count"] == 1


def test_empty_and_whitespace_command_and_error_handling(store):
    # None and empty inputs produce valid hex fingerprints
    fp_empty1 = compute_fingerprint("", "")
    fp_empty2 = compute_fingerprint("   ", "   \t\n   ")
    assert fp_empty1 == fp_empty2
    assert len(fp_empty1) == 16

    fp_none1 = compute_fingerprint(None, None, exit_code=None)
    fp_none2 = compute_fingerprint(None, None, exit_code=1)
    assert len(fp_none1) == 16
    assert len(fp_none2) == 16
    assert fp_none1 != fp_none2

    # Store handles empty/whitespace gracefully
    eid1 = store.add_error("", "")
    eid2 = store.add_error("   ", "   ")
    assert eid1 == eid2
    assert store.get(eid1)["occurrence_count"] == 2

    # Auto capture handles whitespace command gracefully
    inserted, _ = store.auto_capture_tier1("   ", 1, "/tmp")
    assert isinstance(inserted, bool)


# ----------------------------------------------------------- CLI display tests ---


def test_cli_show_displays_occurrences(store):
    eid = store.add_error("docker compose up", "port 80 in use")
    store.add_error("docker compose up", "port 80 in use")
    store.add_error("docker compose up", "port 80 in use")

    result = runner.invoke(app, ["show", str(eid)])
    assert result.exit_code == 0
    assert "occurrences:" in result.output
    assert "3" in result.output
    assert "fingerprint:" in result.output


def test_cli_search_displays_seen_badge(store):
    store.add_error("docker compose up", "port 80 in use")
    store.add_error("docker compose up", "port 80 in use")

    result = runner.invoke(app, ["search", "port 80 in use"])
    assert result.exit_code == 0
    assert "seen 2x" in result.output
