"""Tests for AI-generated error explanations (fixerr.explain, `fixerr
explain`, `fixerr run --explain`, and the dashboard's 'e' keybinding).

The generation backend is mocked throughout — no real Ollama call, same
convention as the rest of the suite.

`fixerr explain` and `fixerr run` generate in a *detached background
process* (see cli._spawn_explain_worker), so an in-process monkeypatch of
the backend can't reach that child interpreter. Those command-level tests
mock ``subprocess.Popen`` instead and assert the worker gets spawned with
the right arguments; the actual generation-and-storage logic is covered
directly against `_explain_worker` (which CliRunner invokes in-process)."""

from __future__ import annotations

from unittest import mock

import pytest
from typer.testing import CliRunner

from fixerr.ai.base import AIUnavailableError
from fixerr.cli import app
from fixerr.explain import generate_explanation
from fixerr.store import Store

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


# ------------------------------------------------------------------- unit ---

def test_generate_explanation_strips_think_block(monkeypatch):
    fake = mock.Mock()
    fake.generate.return_value = "<think>reasoning...</think>Port 5432 was already bound."
    monkeypatch.setattr("fixerr.explain.get_backend", lambda: fake)

    result = generate_explanation("docker compose up", "port already in use")
    assert result == "Port 5432 was already bound."


def test_generate_explanation_returns_none_when_unavailable(monkeypatch):
    fake = mock.Mock()
    fake.generate.side_effect = AIUnavailableError("no backend")
    monkeypatch.setattr("fixerr.explain.get_backend", lambda: fake)

    assert generate_explanation("docker compose up", "port already in use") is None


def test_generate_explanation_returns_none_for_empty_error_text(monkeypatch):
    fake = mock.Mock()
    monkeypatch.setattr("fixerr.explain.get_backend", lambda: fake)
    assert generate_explanation("docker compose up", "") is None
    fake.generate.assert_not_called()


def test_generate_explanation_never_raises_on_unexpected_error(monkeypatch):
    fake = mock.Mock()
    fake.generate.side_effect = RuntimeError("boom")
    monkeypatch.setattr("fixerr.explain.get_backend", lambda: fake)
    assert generate_explanation("docker compose up", "port already in use") is None


# --------------------------------------------------- _explain_worker (in-process) ---
# The actual generate-and-store logic, invoked the same way the detached
# background process invokes it.

def test_explain_worker_generates_and_stores(store, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    monkeypatch.setattr(
        "fixerr.explain.get_backend",
        lambda: mock.Mock(generate=mock.Mock(return_value="Port conflict explanation.")),
    )

    result = runner.invoke(app, ["_explain_worker", str(eid)])
    assert result.exit_code == 0
    assert store.get(eid)["ai_explanation"] == "Port conflict explanation."


def test_explain_worker_unknown_id_is_a_silent_noop(store):
    result = runner.invoke(app, ["_explain_worker", "999999"])
    assert result.exit_code == 0  # never raises — it's a detached background process


def test_explain_worker_backend_unavailable_leaves_field_empty(store, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    monkeypatch.setattr(
        "fixerr.explain.get_backend",
        lambda: mock.Mock(generate=mock.Mock(side_effect=AIUnavailableError("down"))),
    )
    result = runner.invoke(app, ["_explain_worker", str(eid)])
    assert result.exit_code == 0
    assert store.get(eid)["ai_explanation"] is None


def test_explain_worker_hidden_from_help():
    result = runner.invoke(app, ["--help"])
    assert "_explain_worker" not in result.output


# --------------------------------------------------------- fixerr explain ---

def test_cli_explain_spawns_background_worker(store, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    popen = mock.Mock()
    monkeypatch.setattr("subprocess.Popen", popen)

    result = runner.invoke(app, ["explain", str(eid)])
    assert result.exit_code == 0
    assert "background" in result.output.lower()
    popen.assert_called_once()
    args, kwargs = popen.call_args
    cmd = args[0]
    assert cmd[-2:] == ["_explain_worker", str(eid)]
    assert kwargs.get("start_new_session") is True  # detached, survives this process exiting


def test_cli_explain_uses_cache_without_regenerate(store, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_ai_explanation(eid, "cached explanation")
    popen = mock.Mock()
    monkeypatch.setattr("subprocess.Popen", popen)

    result = runner.invoke(app, ["explain", str(eid)])
    assert result.exit_code == 0
    assert "cached explanation" in result.output
    popen.assert_not_called()  # cache hit — no need to spawn anything


def test_cli_explain_regenerate_flag_spawns_worker_again(store, monkeypatch):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_ai_explanation(eid, "stale explanation")
    popen = mock.Mock()
    monkeypatch.setattr("subprocess.Popen", popen)

    result = runner.invoke(app, ["explain", str(eid), "--regenerate"])
    assert result.exit_code == 0
    assert "background" in result.output.lower()
    popen.assert_called_once()


def test_cli_explain_unknown_error_id(store):
    result = runner.invoke(app, ["explain", "999999"])
    assert result.exit_code == 1


# ------------------------------------------------------- fixerr run flag ---

def test_run_explain_default_on_spawns_worker(store, monkeypatch):
    import sys
    # Mocking subprocess.Popen wholesale would also break `run`'s own real
    # subprocess (the wrapped command) and `git rev-parse` inside
    # Store.add_error — target just the spawn call instead.
    spawn = mock.Mock()
    monkeypatch.setattr("fixerr.cli._spawn_explain_worker", spawn)

    result = runner.invoke(
        app, ["run", sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(1)"]
    )
    assert result.exit_code == 1
    assert "background" in result.output.lower()
    spawn.assert_called_once()


def test_run_no_explain_skips_spawning_worker(store, monkeypatch):
    import sys
    spawn = mock.Mock()
    monkeypatch.setattr("fixerr.cli._spawn_explain_worker", spawn)

    result = runner.invoke(
        app,
        ["run", "--no-explain", sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(1)"],
    )
    assert result.exit_code == 1
    assert "background" not in result.output.lower()
    spawn.assert_not_called()
