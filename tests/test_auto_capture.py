"""Tests for Tier 1 (`_auto_capture`, fired by the shell hook) and Tier 2
(`fixerr run`) auto-surfacing.

Embeddings are mocked (deterministic, no network) — same convention as
test_ai_backends.py / test_dashboard.py.
"""

from __future__ import annotations

import time
from unittest import mock

import pytest
from typer.testing import CliRunner

from fixerr.cli import app
from fixerr.store import STATUS_OPEN, Store

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


# --------------------------------------------------------------- tier 1 ---

def test_auto_capture_tier1_dedup_within_window(store):
    inserted1, _ = store.auto_capture_tier1("docker compose up", 1, "/tmp/proj")
    inserted2, _ = store.auto_capture_tier1("docker compose up", 1, "/tmp/proj")
    assert inserted1 is True
    assert inserted2 is False
    assert len(store.list_errors()) == 1


def test_auto_capture_tier1_no_dedup_after_window(store):
    inserted1, _ = store.auto_capture_tier1(
        "docker compose up", 1, "/tmp/proj", dedup_window_seconds=0
    )
    time.sleep(0.05)
    inserted2, _ = store.auto_capture_tier1(
        "docker compose up", 1, "/tmp/proj", dedup_window_seconds=0
    )
    assert inserted1 is True
    assert inserted2 is True
    errors = store.list_errors()
    assert len(errors) == 1
    assert errors[0].occurrence_count == 2


def test_auto_capture_tier1_different_exit_code_not_deduped(store):
    inserted1, _ = store.auto_capture_tier1("docker compose up", 1, "/tmp/proj")
    inserted2, _ = store.auto_capture_tier1("docker compose up", 2, "/tmp/proj")
    assert inserted1 is True
    assert inserted2 is True


def test_auto_capture_tier1_stores_open_status_no_error_text(store):
    store.auto_capture_tier1("docker compose up", 1, "/tmp/proj", git_commit="abc123")
    row = store.list_errors()[0]
    assert row.status == STATUS_OPEN
    assert row.error == ""
    assert row.git_commit == "abc123"


def test_auto_capture_tier1_matches_prior_resolved_same_command(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "killed stale postgres")
    inserted, matches = store.auto_capture_tier1("docker compose up", 1, "/tmp/proj")
    assert inserted is True
    assert len(matches) == 1
    assert matches[0].id == eid
    assert matches[0].fix == "killed stale postgres"


def test_auto_capture_tier1_no_matches_when_no_resolved_history(store):
    inserted, matches = store.auto_capture_tier1("brand-new-command", 1, "/tmp/proj")
    assert inserted is True
    assert matches == []


# -------------------------------------------------- tier 1: CLI surfacing ---

def test_cli_auto_capture_prints_box_when_matches_exist(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "killed stale postgres")

    result = runner.invoke(
        app,
        ["_auto_capture", "--exit-code", "1", "--command", "docker compose up", "--cwd", "/tmp/proj"],
    )
    assert result.exit_code == 0
    assert "killed stale postgres" in result.output


def test_cli_auto_capture_silent_when_no_matches(store):
    result = runner.invoke(
        app,
        ["_auto_capture", "--exit-code", "1", "--command", "brand-new-command", "--cwd", "/tmp/proj"],
    )
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_cli_auto_capture_hidden_from_help():
    result = runner.invoke(app, ["--help"])
    assert "_auto_capture" not in result.output


# --------------------------------------------------------------- tier 2 ---

def test_run_tees_stderr_and_captures_on_failure(store):
    import sys
    result = runner.invoke(
        app,
        ["run", sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(1)"],
    )
    assert result.exit_code == 1
    assert "boom" in result.output  # terminal still saw it (teed, not swallowed)

    errors = store.list_errors()
    assert len(errors) == 1
    assert "boom" in errors[0].error  # and it was captured to the KB


def test_run_stores_nothing_on_success(store):
    import sys
    result = runner.invoke(app, ["run", sys.executable, "-c", "print('ok')"])
    assert result.exit_code == 0
    assert store.list_errors() == []


def test_run_propagates_exact_exit_code(store):
    import sys
    result = runner.invoke(app, ["run", sys.executable, "-c", "import sys; sys.exit(42)"])
    assert result.exit_code == 42


def test_run_surfaces_similar_past_resolved_error(store):
    import sys
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "killed stale postgres")

    result = runner.invoke(
        app,
        ["run", sys.executable, "-c", "import sys; sys.stderr.write('port already in use\\n'); sys.exit(1)"],
    )
    assert result.exit_code == 1
    assert "killed stale postgres" in result.output


# ------------------------------------------------- [capture] policy gate ---

def test_ignore_commands_default_blocks_cd(store):
    """`cd` is in the default ignore_commands — must never reach the DB."""
    result = runner.invoke(
        app, ["_auto_capture", "--exit-code", "1", "--command", "cd /tmp", "--cwd", "/tmp"]
    )
    assert result.exit_code == 0
    assert store.list_errors() == []


def test_ignore_commands_is_a_prefix_match(store):
    """"git status" in ignore_commands also excludes "git status --short"."""
    result = runner.invoke(
        app,
        ["_auto_capture", "--exit-code", "1", "--command", "git status --short", "--cwd", "/tmp"],
    )
    assert result.exit_code == 0
    assert store.list_errors() == []


def test_non_ignored_command_still_captured(store):
    result = runner.invoke(
        app, ["_auto_capture", "--exit-code", "1", "--command", "docker compose up", "--cwd", "/tmp"]
    )
    assert result.exit_code == 0
    assert len(store.list_errors()) == 1


def test_auto_capture_false_blocks_everything(store):
    from fixerr import config as cfg

    cfg.set_value("capture.auto_capture", "false")
    result = runner.invoke(
        app, ["_auto_capture", "--exit-code", "1", "--command", "docker compose up", "--cwd", "/tmp"]
    )
    assert result.exit_code == 0
    assert store.list_errors() == []


def test_min_exit_code_filters_low_exit_codes(store):
    from fixerr import config as cfg

    cfg.set_value("capture.min_exit_code", "2")
    result = runner.invoke(
        app, ["_auto_capture", "--exit-code", "1", "--command", "docker compose up", "--cwd", "/tmp"]
    )
    assert result.exit_code == 0
    assert store.list_errors() == []

    result = runner.invoke(
        app, ["_auto_capture", "--exit-code", "2", "--command", "docker compose up", "--cwd", "/tmp"]
    )
    assert len(store.list_errors()) == 1


def test_quiet_captures_silently(store):
    from fixerr import config as cfg

    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "killed stale postgres")
    cfg.set_value("capture.quiet", "true")

    result = runner.invoke(
        app, ["_auto_capture", "--exit-code", "1", "--command", "docker compose up", "--cwd", "/tmp"]
    )
    assert result.exit_code == 0
    assert result.output.strip() == ""  # nothing printed...
    assert len(store.list_errors()) == 2  # ...but it was still captured


def test_surface_threshold_zero_shows_box_with_no_matches(store):
    from fixerr import config as cfg

    cfg.set_value("capture.surface_threshold", "0")
    result = runner.invoke(
        app, ["_auto_capture", "--exit-code", "1", "--command", "brand-new-cmd", "--cwd", "/tmp"]
    )
    assert result.exit_code == 0
    assert "No past fixes yet" in result.output


def test_ignore_patterns_regex_blocks_matching_command():
    # ignore_patterns.patterns is a list; `config set` only writes scalars
    # (see config._dumps), so this exercises the matcher directly against a
    # config dict shaped the way a hand-edited toml array would parse.
    from fixerr.cli import _command_is_ignored

    capture_cfg = {"ignore_patterns": {"patterns": ["^vim "]}}
    assert _command_is_ignored("vim myfile.py", capture_cfg) is True
    assert _command_is_ignored("nvim myfile.py", capture_cfg) is False


# ------------------------------------------------------- config coercion ---

def test_config_set_bool_round_trips_as_real_boolean(tmp_path, monkeypatch):
    """Regression test: `config set` used to store "false" as a truthy string."""
    monkeypatch.setenv("fixerr_CONFIG", str(tmp_path / "config.toml"))
    from fixerr import config as cfg

    cfg.set_value("capture.auto_capture", "false")
    loaded = cfg.load_config()
    assert loaded["capture"]["auto_capture"] is False  # real bool, not the string "false"

    cfg.set_value("capture.min_exit_code", "2")
    loaded = cfg.load_config()
    assert loaded["capture"]["min_exit_code"] == 2
    assert isinstance(loaded["capture"]["min_exit_code"], int)


# ------------------------------------------------------------- shell-init ---

def test_shell_init_pwsh_default():
    result = runner.invoke(app, ["shell-init", "pwsh"])
    assert result.exit_code == 0
    out = result.output
    assert "function global:prompt" in out
    assert "$__succeeded = $?" in out
    assert "$__lastexitcode_now = $LASTEXITCODE" in out
    assert "Get-History -Count 1" in out
    assert "$__wasInterrupted" in out
    assert "Start-Process -FilePath 'fixerr'" in out
    assert "$global:__fixerr_min_exit_code = 1" in out
    assert "'git status'" in out
    assert "'cd'" in out


def test_shell_init_pwsh_with_custom_config(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[capture]\nmin_exit_code = 3\nignore_commands = ["git status", "cd", "ls"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("fixerr_CONFIG", str(config_file))

    result = runner.invoke(app, ["shell-init", "pwsh"])
    assert result.exit_code == 0
    out = result.output
    assert "$global:__fixerr_min_exit_code = 3" in out
    assert "$global:__fixerr_ignore = @('git status','cd','ls')" in out


def test_shell_init_pwsh_quote_escaping(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[capture]\nignore_commands = ["it\'s a test", "foo\\"bar"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("fixerr_CONFIG", str(config_file))

    result = runner.invoke(app, ["shell-init", "pwsh"])
    assert result.exit_code == 0
    out = result.output
    # PowerShell escapes single quotes by doubling: 'it''s a test'
    assert "'it''s a test'" in out


def test_shell_init_unsupported_shell():
    result = runner.invoke(app, ["shell-init", "fish"])
    assert result.exit_code == 1
    assert "unsupported shell" in result.output.lower() or "unsupported shell" in (result.stderr if hasattr(result, "stderr") else "")

