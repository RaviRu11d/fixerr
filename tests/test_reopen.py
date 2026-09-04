"""Tests for reverting a resolved/won't-fix error back to open
(Store.reopen, ErnestClient.reopen, `fixerr reopen`)."""

from __future__ import annotations

from unittest import mock

import pytest
from typer.testing import CliRunner

from fixerr.cli import app
from fixerr.client import ErnestClient
from fixerr.store import STATUS_OPEN, STATUS_RESOLVED, STATUS_WONT_FIX, Store

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


# ------------------------------------------------------------------- store ---

def test_reopen_reverts_resolved_to_open_and_clears_resolved_at(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "killed stale postgres")
    assert store.get(eid)["status"] == STATUS_RESOLVED

    assert store.reopen(eid) is True
    row = store.get(eid)
    assert row["status"] == STATUS_OPEN
    assert row["resolved_at"] is None


def test_reopen_keeps_fix_text(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "killed stale postgres")
    store.reopen(eid)
    assert store.get(eid)["fix_text"] == "killed stale postgres"


def test_reopen_reverts_wont_fix_to_open(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.dismiss(eid)
    assert store.get(eid)["status"] == STATUS_WONT_FIX

    assert store.reopen(eid) is True
    assert store.get(eid)["status"] == STATUS_OPEN


def test_reopen_unknown_id_returns_false(store):
    assert store.reopen(999999) is False


# ------------------------------------------------------------------ client ---

def test_client_reopen_delegates_to_store(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "some fix")
    client = ErnestClient()
    assert client.reopen(eid) is True
    assert store.get(eid)["status"] == STATUS_OPEN


# --------------------------------------------------------------------- CLI ---

def test_cli_reopen_command(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "some fix")

    result = runner.invoke(app, ["reopen", str(eid)])
    assert result.exit_code == 0
    assert "reopened" in result.output.lower()
    assert store.get(eid)["status"] == STATUS_OPEN


def test_cli_reopen_unknown_id(store):
    result = runner.invoke(app, ["reopen", "999999"])
    assert result.exit_code == 1
