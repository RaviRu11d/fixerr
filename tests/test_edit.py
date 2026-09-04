"""Tests for `fixerr edit` ($EDITOR-based fix editing).

$EDITOR is pointed at a tiny python script (no real terminal editor involved)
that deterministically rewrites the temp file, so these run headless and
cross-platform.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest
from typer.testing import CliRunner

from fixerr.cli import app
from fixerr.store import STATUS_OPEN, STATUS_RESOLVED, Store

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


def _fake_editor(tmp_path, py_code: str) -> str:
    """Write a python script standing in for $EDITOR and return its command."""
    path = tmp_path / "fake_editor.py"
    path.write_text(py_code, encoding="utf-8")
    return f'"{sys.executable}" "{path}"'


def test_edit_appends_multiline_fix_and_resolves(store, tmp_path, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    editor = _fake_editor(
        tmp_path,
        "import sys\nwith open(sys.argv[1], 'a', encoding='utf-8') as f:\n    f.write('line one\\nline two\\n')",
    )
    monkeypatch.setenv("EDITOR", editor)

    result = runner.invoke(app, ["edit", str(eid)])
    assert result.exit_code == 0
    assert "updated" in result.output.lower()

    row = store.get(eid)
    assert row["status"] == STATUS_RESOLVED
    assert row["fix_text"] == "line one\nline two"


def test_edit_strips_comment_lines(store, tmp_path, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    editor = _fake_editor(
        tmp_path,
        "import sys\nwith open(sys.argv[1], 'a', encoding='utf-8') as f:\n    f.write('# not a fix\\nreal fix line\\n')",
    )
    monkeypatch.setenv("EDITOR", editor)

    runner.invoke(app, ["edit", str(eid)])
    assert store.get(eid)["fix_text"] == "real fix line"


def test_edit_no_changes_leaves_fix_untouched(store, tmp_path, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "original fix")
    editor = _fake_editor(tmp_path, "import sys\npass")  # no-op, file unchanged
    monkeypatch.setenv("EDITOR", editor)

    result = runner.invoke(app, ["edit", str(eid)])
    assert result.exit_code == 0
    assert "no changes" in result.output.lower()
    assert store.get(eid)["fix_text"] == "original fix"


def test_edit_empty_fix_is_a_noop(store, tmp_path, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    editor = _fake_editor(
        tmp_path,
        "import sys\nwith open(sys.argv[1], 'w', encoding='utf-8') as f:\n    f.write('')",
    )  # truncate the file entirely
    monkeypatch.setenv("EDITOR", editor)

    result = runner.invoke(app, ["edit", str(eid)])
    assert result.exit_code == 0
    assert "empty" in result.output.lower()
    row = store.get(eid)
    assert row["status"] == STATUS_OPEN  # never got resolved
    assert not (row["fix_text"] or "")


def test_edit_prefills_existing_fix(store, tmp_path, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "existing fix text")
    captured = tmp_path / "captured.txt"
    escaped_captured = str(captured).replace("\\", "\\\\")
    editor = _fake_editor(
        tmp_path,
        f"import sys, shutil\nshutil.copy(sys.argv[1], r'{escaped_captured}')",
    )
    monkeypatch.setenv("EDITOR", editor)

    runner.invoke(app, ["edit", str(eid)])
    text = captured.read_text(encoding="utf-8")
    assert "existing fix text" in text
    assert "docker compose up" in text  # command shown as context
    assert "port already in use" in text  # error shown as context


def test_edit_unknown_error_id(store):
    result = runner.invoke(app, ["edit", "999999"])
    assert result.exit_code == 1


def test_edit_editor_with_arguments_is_split_correctly(store, tmp_path, monkeypatch):
    # $EDITOR can be "vim -n" / "code --wait" etc — must not be treated as
    # one literal (nonexistent) executable name.
    editor_script = _fake_editor(
        tmp_path,
        "import sys\nwith open(sys.argv[2], 'a', encoding='utf-8') as f:\n    f.write('fixed via args\\n')",
    )
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    # simulate "<script> --flag <path>" by having the script take the path as $2
    monkeypatch.setenv("EDITOR", f"{editor_script} --flag")

    runner.invoke(app, ["edit", str(eid)])
    assert store.get(eid)["fix_text"] == "fixed via args"


def test_edit_missing_editor_reports_a_clear_error(store, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    monkeypatch.setenv("EDITOR", "/nonexistent/definitely-not-an-editor")

    result = runner.invoke(app, ["edit", str(eid)])
    assert result.exit_code == 1
    assert "editor" in result.output.lower()
