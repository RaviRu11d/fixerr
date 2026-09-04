"""``fixerr`` command-line interface.

Exposes the error/fix knowledge base (``capture`` / ``resolve`` / ``search`` /
``show``), configuration (``config set`` / ``config show``), a ``doctor``
provider health check, and the interactive ``dashboard`` TUI. devnest mounts
this whole app under ``devnest errors``.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console

from . import config as cfg
from .ai import get_backend, get_embed_backend
from .client import ErnestClient
from .store import Match

app = typer.Typer(help="Local-first error/fix memory.", no_args_is_help=True)
config_app = typer.Typer(help="View and change fixerr configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")
console = Console()
console_err = Console(stderr=True)


# --------------------------------------------------------------- KB commands ---

def _read_stdin() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read()
    console.print("[dim]Paste the error output, then press Ctrl-D:[/dim]")
    return sys.stdin.read()


@app.command("capture")
def capture(
    command: Optional[str] = typer.Option(None, "--command", "-c", help="The failing command."),
    exit_code: Optional[int] = typer.Option(None, "--exit-code", help="Exit code of the failing command."),
) -> None:
    """Capture a failing command + its error output (read from stdin) as unresolved."""
    if command is None:
        command = typer.prompt("Failing command")
    error_text = _read_stdin().strip()
    if not error_text:
        console.print("[yellow]No error text provided; nothing captured.[/yellow]")
        raise typer.Exit(code=1)
    error_id = ErnestClient().record(command, error_text, exit_code=exit_code)
    console.print(
        f"[green]Captured as error #{error_id}[/green] (unresolved). "
        f"When fixed: [cyan]fixerr resolve {error_id} \"...\"[/cyan]."
    )


@app.command("resolve")
def resolve(
    error_id: int = typer.Argument(..., help="Error id to resolve."),
    fix: str = typer.Argument(..., help="How you fixed it."),
) -> None:
    """Record the fix for a captured error and mark it resolved."""
    if ErnestClient().resolve(error_id, fix):
        console.print(f"[green]Error #{error_id} marked resolved.[/green]")
    else:
        console.print(f"[red]No error #{error_id}.[/red]")
        raise typer.Exit(code=1)


@app.command("edit")
def edit(error_id: int = typer.Argument(..., help="Error id.")) -> None:
    """Edit an error's fix text in $EDITOR (like `git commit -e`).

    Opens the current fix (if any) in your editor, with the error text shown
    as commented-out context. Save and exit to apply; leave it unchanged (or
    empty) to cancel. Marks the error resolved, same as `fixerr resolve`.
    """
    import os
    import shlex
    import subprocess
    import tempfile

    from .store import Store

    store = Store()
    row = store.get(error_id)
    if row is None:
        console.print(f"[red]No error #{error_id}.[/red]")
        raise typer.Exit(code=1)

    existing = (row["fix_text"] or "").strip()
    header = (
        f"# Editing the fix for error #{error_id}: {row['failing_command'] or '?'}\n"
        f"# Lines starting with '#' are stripped. Save and exit to apply;\n"
        f"# leave this empty/unchanged to cancel.\n"
        f"#\n"
        f"# Error (redacted):\n"
    )
    for line in (row["error_text_redacted"] or "").splitlines()[:10]:
        header += f"# {line}\n"
    header += "#\n"

    editor_cmd = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"

    fd, path = tempfile.mkstemp(prefix=f"fixerr-fix-{error_id}-", suffix=".md")
    try:
        with os.fdopen(fd, "w") as tf:
            tf.write(header)
            tf.write(existing)

        try:
            subprocess.run(shlex.split(editor_cmd) + [path], check=False)
        except OSError as exc:
            console.print(
                f"[red]Couldn't launch editor {editor_cmd!r}: {exc}. "
                f"Set $EDITOR to a valid command.[/red]"
            )
            raise typer.Exit(code=1) from None

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    finally:
        os.unlink(path)

    new_fix = "\n".join(
        line for line in content.splitlines() if not line.strip().startswith("#")
    ).strip()

    if not new_fix:
        console.print("[yellow]Empty fix — nothing changed.[/yellow]")
        return
    if new_fix == existing:
        console.print("[dim]No changes.[/dim]")
        return

    store.set_fix(error_id, new_fix)
    console.print(f"[green]Error #{error_id} fix updated.[/green]")


@app.command("dismiss")
def dismiss(error_id: int = typer.Argument(..., help="Error id to mark won't-fix.")) -> None:
    """Mark an error as won't-fix (excluded from future auto-surfacing)."""
    if ErnestClient().dismiss(error_id):
        console.print(f"[yellow]Error #{error_id} marked won't-fix.[/yellow]")
    else:
        console.print(f"[red]No error #{error_id}.[/red]")
        raise typer.Exit(code=1)


@app.command("reopen")
def reopen(
    error_id: int = typer.Argument(..., help="Error id to reopen (undo resolve/dismiss)."),
) -> None:
    """Revert a resolved or won't-fix error back to open. Keeps its fix text."""
    if ErnestClient().reopen(error_id):
        console.print(f"[yellow]Error #{error_id} reopened.[/yellow]")
    else:
        console.print(f"[red]No error #{error_id}.[/red]")
        raise typer.Exit(code=1)


# ------------------------------------------------------ automatic surfacing ---
# Tier 1: a fully automatic shell hook (see `shell-init`) fires this on every
# non-zero exit. It must stay fast — no redaction, no embedding, no Ollama —
# see Store.auto_capture_tier1 for the rationale.

def _relative_time(iso_ts: Optional[str]) -> str:
    if not iso_ts:
        return "?"
    try:
        ts = datetime.fromisoformat(iso_ts)
    except ValueError:
        return "?"
    now = datetime.now(timezone.utc) if ts.tzinfo else datetime.now()
    secs = (now - ts).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    if secs < 86400 * 7:
        return f"{int(secs // 86400)}d ago"
    return f"{int(secs // (86400 * 7))}w ago"


def _surface_panel(command: str, matches: List[Match]):
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    if not matches:
        # Only reachable with surface_threshold=0 — "show even with no
        # matches" per [capture] surface_threshold's documented behavior.
        body.append(f"No past fixes yet for: {command}", style="dim")
        return Panel(body, title="fixerr", border_style="cyan", expand=False)
    body.append(f"Similar past errors found for: {command}\n\n", style="bold")
    for m in matches:
        label = (m.error or "").splitlines()[0][:60] if m.error else command
        body.append("● ", style="cyan")
        body.append(f"{label} (resolved {_relative_time(m.resolved_at)})\n")
        if m.fix:
            fix_line = m.fix.splitlines()[0]
            body.append(f"  Fix: {fix_line[:70]}\n", style="dim")
        body.append("\n")
    ids = "  ".join(f"#{m.id}" for m in matches)
    body.append(
        f"fixerr show <id>  for full details · fixerr resolve <id> \"...\" "
        f"to update  [{ids}]",
        style="dim italic",
    )
    return Panel(body, title="fixerr", border_style="cyan", expand=False)


def _as_bool(value: Any, default: bool) -> bool:
    """Defensive bool coercion for config values.

    ``config.set_value`` now stores real TOML booleans, but this also guards
    against a hand-edited config with a quoted ``"false"`` string, which
    would otherwise be truthy and silently defeat the gate it's meant to be.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "")
    return bool(value)


def _command_is_ignored(command: str, capture_cfg: Dict[str, Any]) -> bool:
    """True if ``command`` matches an ignore_commands prefix or ignore_patterns regex.

    ``ignore_commands`` is a prefix match (mirrors the shell hook's own
    ``[[ "$cmd" == "$skip"* ]]`` check) so e.g. "git status" also excludes
    "git status --short". A malformed user-supplied regex is skipped rather
    than raised — a typo in config must never break auto-capture entirely.
    """
    for prefix in capture_cfg.get("ignore_commands", []) or []:
        if command.startswith(prefix):
            return True
    patterns = (capture_cfg.get("ignore_patterns") or {}).get("patterns", [])
    for pattern in patterns or []:
        try:
            if re.search(pattern, command):
                return True
        except re.error:
            continue
    return False


@app.command("_auto_capture", hidden=True)
def auto_capture(
    exit_code: int = typer.Option(..., "--exit-code"),
    command: str = typer.Option(..., "--command"),
    cwd: str = typer.Option(..., "--cwd"),
    git_commit: str = typer.Option("", "--git-commit"),
) -> None:
    """Internal: fired by the shell hook on every failed command. Not for direct use.

    Runs backgrounded from an interactive shell prompt — it must never leak a
    raw traceback into the user's terminal, so every failure mode here is
    swallowed rather than raised (mirrors ErnestClient.surface()'s contract).

    This is the *authoritative* policy gate — the shell hook also bakes in a
    fast-path check (skip forking Python at all for obviously-boring
    commands), but this check always runs too, so it stays correct even
    against a stale/unregenerated shell script or a direct manual invocation.
    """
    try:
        capture_cfg = cfg.load_config().get("capture", {})
        if not _as_bool(capture_cfg.get("auto_capture"), True):
            return
        if exit_code < int(capture_cfg.get("min_exit_code", 1)):
            return
        if _command_is_ignored(command, capture_cfg):
            return

        from .store import Store

        inserted, matches = Store().auto_capture_tier1(
            failing_command=command,
            exit_code=exit_code,
            cwd=cwd,
            git_commit=git_commit or None,
        )
        if not inserted:
            return
        if len(matches) < int(capture_cfg.get("surface_threshold", 1)):
            return
        if _as_bool(capture_cfg.get("quiet"), False):
            return
        console_err.print(_surface_panel(command, matches))
    except Exception:  # noqa: BLE001 - a background hook must never crash visibly
        return


# ------------------------------------------------------------------ shell hook ---
# Registers the Tier 1 auto-capture on every non-zero exit. Command detection is
# native in zsh (preexec/TRAPZERR); bash has no equivalent hook point, so it
# falls back to a `trap ... DEBUG` approximation, which can occasionally
# misattribute the captured command inside pipelines/compound commands — noted
# inline below.
#
# Backgrounding via `( ... & )` (a subshell that itself exits immediately after
# launching the background job) detaches the job from the parent shell's job
# table, so no "[1] Done" notification appears — this is why stderr is *not*
# redirected to /dev/null here: `_auto_capture` prints its surface box to
# stderr, and it already swallows its own errors internally, so there is
# nothing left that redirecting would need to hide.

def _shell_array_literal(items: List[str]) -> str:
    """Render a Python string list as a single-quoted, space-separated shell
    array body (works for both zsh and bash array literal syntax)."""
    return " ".join("'" + item.replace("'", "'\\''") + "'" for item in items)


def _pwsh_array_literal(items: List[str]) -> str:
    """Render a Python string list as a PowerShell array literal, e.g.
    @('git status','cd'). PowerShell single-quoted strings escape an
    embedded quote by doubling it (unlike bash's '\\'' trick above)."""
    if not items:
        return "@()"
    return "@(" + ",".join("'" + item.replace("'", "''") + "'" for item in items) + ")"

# The ignore-array and min-exit-code tokens below are plain-text sentinels,
# not str.format() placeholders — the script itself is full of literal `{`/`}`
# (function bodies) and `$name` references, either of which would collide
# with a real templating engine. shell_init() replaces them with plain
# .replace() calls instead. Baking config into the printed script means the
# shell-level check is a *fast-path only* — it reflects config as of the last
# `eval "$(fixerr shell-init ...)"`, and goes stale until re-sourced after a
# config change. `_auto_capture` re-reads config fresh on every invocation
# regardless, so correctness never depends on the shell script being current.

_ZSH_HOOK = r"""# fixerr Tier 1: auto-capture every failed command.
__fixerr_last_cmd=""
__fixerr_preexec() { __fixerr_last_cmd="$1" }

__fixerr_ignore=(__fixerr_IGNORE_ARRAY__)
__fixerr_min_exit_code=__fixerr_MIN_EXIT_CODE__

TRAPZERR() {
  local __ec=$?
  local __cmd="$__fixerr_last_cmd"
  [[ -z "$__cmd" ]] && return
  [[ "$__cmd" == fixerr* ]] && return
  (( __ec < __fixerr_min_exit_code )) && return
  local __skip
  for __skip in "${__fixerr_ignore[@]}"; do
    [[ "$__cmd" == "$__skip"* ]] && return
  done
  ( command fixerr _auto_capture \
      --exit-code "$__ec" \
      --command "$__cmd" \
      --cwd "$PWD" \
      --git-commit "$(git rev-parse --short HEAD 2>/dev/null)" \
      & )
  return $__ec
}

autoload -Uz add-zsh-hook
add-zsh-hook preexec __fixerr_preexec
"""

_BASH_HOOK = r"""# fixerr Tier 1: auto-capture every failed command.
# Note: bash has no native preexec/precmd hook pair (unlike zsh), so this uses
# a DEBUG-trap approximation. It can occasionally capture the wrong command
# inside pipelines or `&&`/`;`-chained compound commands, since DEBUG fires
# before every simple command, not once per typed line. For exact behavior,
# install bash-preexec (https://github.com/rcaloras/bash-preexec) — if
# ~/.bash-preexec.sh is present, it's used instead of the approximation below.
#
# The exit code is always captured via an ERR trap (not raw `$?` inside a
# precmd function) — with bash-preexec, other registered precmd_functions may
# run before ours and clobber `$?` before we'd get to read it.
__fixerr_last_ec=0
trap '__fixerr_last_ec=$?' ERR

__fixerr_ignore=(__fixerr_IGNORE_ARRAY__)
__fixerr_min_exit_code=__fixerr_MIN_EXIT_CODE__

__fixerr_should_skip() {
  local __cmd="$1" __skip
  for __skip in "${__fixerr_ignore[@]}"; do
    [[ "$__cmd" == "$__skip"* ]] && return 0
  done
  return 1
}

if [[ -f "$HOME/.bash-preexec.sh" ]]; then
  source "$HOME/.bash-preexec.sh"
  __fixerr_preexec() { __fixerr_last_cmd="$1"; }
  preexec_functions+=(__fixerr_preexec)
  __fixerr_precmd() {
    local __ec=$__fixerr_last_ec
    __fixerr_last_ec=0
    [[ $__ec -eq 0 ]] && return
    (( __ec < __fixerr_min_exit_code )) && return
    local __cmd="$__fixerr_last_cmd"
    [[ -z "$__cmd" ]] && return
    [[ "$__cmd" == fixerr* ]] && return
    __fixerr_should_skip "$__cmd" && return
    ( command fixerr _auto_capture \
        --exit-code "$__ec" \
        --command "$__cmd" \
        --cwd "$PWD" \
        --git-commit "$(git rev-parse --short HEAD 2>/dev/null)" \
        & )
  }
  precmd_functions+=(__fixerr_precmd)
else
  __fixerr_last_cmd=""
  # $BASH_COMMAND for the DEBUG trap firing on PROMPT_COMMAND's own
  # sub-commands (including this precmd function itself) would otherwise
  # clobber the real captured command right before precmd reads it — exclude
  # our own internal names inline rather than trying to track that with a
  # flag (which has its own ordering race against the first DEBUG firing).
  trap '[[ "$BASH_COMMAND" != __fixerr_* ]] && __fixerr_last_cmd=$BASH_COMMAND' DEBUG

  __fixerr_precmd() {
    local __ec=$__fixerr_last_ec
    local __cmd="$__fixerr_last_cmd"
    __fixerr_last_ec=0
    [[ $__ec -eq 0 ]] && return
    (( __ec < __fixerr_min_exit_code )) && return
    [[ -z "$__cmd" ]] && return
    [[ "$__cmd" == fixerr* ]] && return
    [[ "$__cmd" == __fixerr_precmd* ]] && return
    __fixerr_should_skip "$__cmd" && return
    ( command fixerr _auto_capture \
        --exit-code "$__ec" \
        --command "$__cmd" \
        --cwd "$PWD" \
        --git-commit "$(git rev-parse --short HEAD 2>/dev/null)" \
        & )
  }
  PROMPT_COMMAND="__fixerr_precmd${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
fi
"""


_PWSH_HOOK = r"""# fixerr Tier 1: auto-capture every failed command.
# PowerShell has no native preexec/precmd hook pair (unlike zsh); this
# overrides the `prompt` function, PowerShell's standard hook point (same
# mechanism posh-git / oh-my-posh use), which runs once after each command
# finishes and once before the next prompt is drawn.
#
# Failure signal: `$?` alone, NOT `$LASTEXITCODE`. `$?` is reliably updated
# after every command — PowerShell-native cmdlets AND external processes
# (python, npm, git, ...) both set it correctly. `$LASTEXITCODE` is only
# ever set by external processes; it stays stale after a native cmdlet
# fails, and is unset entirely in a fresh session. Gating on it alone (a
# tempting shortcut, since it feels closer to bash's $?) silently drops
# every pure-PowerShell error — checked here, verified against a real
# ModuleNotFoundError from an external `python` call before relying on it.
if (-not $global:__fixerr_hook_installed) {
    $global:__fixerr_hook_installed = $true

    $global:__fixerr_original_prompt = if (Test-Path Function:\prompt) {
        $function:prompt
    } else {
        { "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) " }
    }

    $global:__fixerr_ignore = __fixerr_IGNORE_ARRAY__
    $global:__fixerr_min_exit_code = __fixerr_MIN_EXIT_CODE__
    $global:__fixerr_prev_lastexitcode = $null

    # Escapes one value for safe inclusion in a manually-built Windows
    # command-line string (wraps in quotes, doubles embedded quotes) — see
    # the note above the argument-building block below for why this is
    # built by hand rather than via Start-Process -ArgumentList's array form.
    function global:__fixerr_quote_arg($s) {
        if ($null -eq $s) { $s = '' }
        '"' + ($s -replace '"', '""') + '"'
    }

    function global:prompt {
        $__succeeded = $?
        $__lastexitcode_now = $LASTEXITCODE
        $__lastEntry = Get-History -Count 1

        # Ctrl+C sets the history entry's own ExecutionStatus to 'Stopped' —
        # PowerShell's direct, built-in signal for "the user interrupted
        # this." Earlier drafts inferred this indirectly via $Error
        # (checking for a PipelineStoppedException), but $Error does not
        # populate reliably for a raw console Ctrl+C in every host —
        # confirmed empirically: it still logged interrupted commands as
        # real errors. ExecutionStatus is a direct field on the history
        # entry itself, not an inference, so it doesn't have that gap.
        $__wasInterrupted = $__lastEntry -and $__lastEntry.ExecutionStatus -eq 'Stopped'

        if ((-not $__succeeded) -and (-not $__wasInterrupted)) {
            $__cmd = $__lastEntry.CommandLine

            if ($__cmd) {
                $__skip = $false
                if ($__cmd -like 'fixerr*') { $__skip = $true }
                foreach ($__ignored in $global:__fixerr_ignore) {
                    if ($__cmd.StartsWith($__ignored)) { $__skip = $true; break }
                }

                if (-not $__skip) {
                    # $LASTEXITCODE only reflects THIS failure if it changed
                    # since we last looked; otherwise it's a stale value left
                    # over from an earlier external command, and this failure
                    # is a native cmdlet error that never touched it.
                    $__ec = if ($null -ne $__lastexitcode_now -and
                                $__lastexitcode_now -ne $global:__fixerr_prev_lastexitcode) {
                        $__lastexitcode_now
                    } else {
                        1
                    }

                    if ($__ec -ge $global:__fixerr_min_exit_code) {
                        $__cwd = (Get-Location).Path
                        $__gitCommit = ''
                        try {
                            $__gitCommit = (git rev-parse --short HEAD 2>$null)
                            if (-not $__gitCommit) { $__gitCommit = '' }
                        } catch { $__gitCommit = '' }

                        # Start-Process -ArgumentList as an array does not
                        # reliably preserve arguments containing spaces or
                        # embedded quotes (e.g. the captured command itself,
                        # like `python -c "import x"`) — they can get split
                        # into separate tokens before fixerr ever sees them,
                        # which broke exactly this case in testing. Build one
                        # explicitly-quoted command-line string instead.
                        $__argsStr = "_auto_capture --exit-code $__ec " +
                            "--command $(__fixerr_quote_arg $__cmd) " +
                            "--cwd $(__fixerr_quote_arg $__cwd)"
                        if ($__gitCommit) {
                            $__argsStr += " --git-commit $(__fixerr_quote_arg $__gitCommit)"
                        }

                        try {
                            Start-Process -FilePath 'fixerr' -ArgumentList $__argsStr `
                                -NoNewWindow -PassThru -ErrorAction Stop | Out-Null
                        } catch {
                            # fixerr not on PATH, or spawn failed — never
                            # let auto-capture break the prompt itself.
                        }
                    }
                }
            }
        }

        $global:__fixerr_prev_lastexitcode = $__lastexitcode_now

        & $global:__fixerr_original_prompt
    }
}
"""



@app.command("shell-init")
def shell_init(
    shell: str = typer.Argument(..., help="Shell to generate the hook for: 'zsh', 'bash', or 'pwsh'."),
) -> None:
    """Print a shell hook that auto-captures every failed command (Tier 1).

    Add to your shell rc file:

    \b
      eval "$(fixerr shell-init zsh)"   # in ~/.zshrc
      eval "$(fixerr shell-init bash)"  # in ~/.bashrc
      fixerr shell-init pwsh | Out-String | Invoke-Expression   # in $PROFILE

    Output is plain text meant for `eval` (or `Invoke-Expression` on
    PowerShell) — never printed via Rich, so nothing here can inject ANSI
    codes or wrap lines into the generated script. Bakes in the current
    [capture] ignore_commands / min_exit_code as a fast-path (re-run this
    after changing those in config.toml to pick up the change).
    """
    shell_name = shell.strip().lower()
    if shell_name not in ("zsh", "bash", "pwsh"):
        console_err.print(f"[red]Unsupported shell: {shell!r}. Use 'zsh', 'bash', or 'pwsh'.[/red]")
        raise typer.Exit(code=1)

    capture_cfg = cfg.load_config().get("capture", {})
    try:
        min_exit_code = int(capture_cfg.get("min_exit_code", 1))
    except (TypeError, ValueError):
        min_exit_code = 1

    if shell_name == "pwsh":
        array_literal = _pwsh_array_literal(capture_cfg.get("ignore_commands", []) or [])
        script = _PWSH_HOOK.replace("__fixerr_IGNORE_ARRAY__", array_literal).replace(
            "__fixerr_MIN_EXIT_CODE__", str(min_exit_code)
        )
        print(script)
        return

    array_literal = _shell_array_literal(capture_cfg.get("ignore_commands", []) or [])
    template = _ZSH_HOOK if shell_name == "zsh" else _BASH_HOOK
    script = template.replace("__fixerr_IGNORE_ARRAY__", array_literal).replace(
        "__fixerr_MIN_EXIT_CODE__", str(min_exit_code)
    )
    print(script)


# ------------------------------------------------------------------- run ---
# Tier 2: `fixerr run <command>` wraps a command, tees its stderr (terminal
# still sees everything, unchanged), and on failure stores the *full* redacted
# stderr with an embedding — unlike Tier 1's command-only auto-capture, this
# gives real semantic search against the KB, not just exact command matches.

def _tee_run(command_parts: List[str]) -> tuple[int, str]:
    """Run ``command_parts``, streaming stdout/stderr to the terminal as normal
    while also capturing stderr for storage. Returns (returncode, stderr_text)."""
    import io
    import subprocess
    import threading

    stderr_buffer = io.StringIO()
    proc = subprocess.Popen(command_parts, stderr=subprocess.PIPE, stdout=None)

    def tee_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            decoded = line.decode(errors="replace")
            sys.stderr.write(decoded)
            stderr_buffer.write(decoded)
        proc.stderr.close()

    t = threading.Thread(target=tee_stderr)
    t.start()
    proc.wait()
    t.join()
    return proc.returncode, stderr_buffer.getvalue()


def _run_result_panel(
    error_id: int, returncode: int, matches: List[Match], explaining: bool = False
):
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    body.append(f"Stored as #{error_id}", style="bold")
    body.append(f"  (fixerr resolve {error_id} \"...\" to add a fix)\n")
    if explaining:
        body.append(
            f"AI explanation generating in the background — check `fixerr show {error_id}` "
            f"shortly.\n",
            style="dim",
        )
    if matches:
        body.append("\nSimilar past errors:\n", style="bold")
        for m in matches:
            label = (m.error or "").splitlines()[0][:50] if m.error else (m.failing_command or "?")
            body.append(f"● {m.score:.0%} — ", style="cyan")
            body.append(f"{label} ({m.failing_command or '?'}, {_relative_time(m.resolved_at or m.timestamp)})\n")
            if m.fix:
                fix_line = m.fix.splitlines()[0]
                body.append(f"  Fix: {fix_line[:70]}\n", style="dim")
    return Panel(body, title=f"fixerr captured error (exit {returncode})", border_style="cyan", expand=False)


@app.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    help="Run a command, capture stderr on failure, and surface similar past errors.",
)
def run_wrapped(
    command_parts: List[str] = typer.Argument(
        ..., help="The command to run, e.g. `fixerr run docker compose up`."
    ),
    explain_flag: bool = typer.Option(
        True,
        "--explain/--no-explain",
        help="Generate an AI explanation in the background. On by default; adds no latency here.",
    ),
) -> None:
    if not command_parts:
        console_err.print("[red]No command given.[/red]")
        raise typer.Exit(code=2)

    returncode, stderr_text = _tee_run(command_parts)

    if returncode == 0:
        raise typer.Exit(code=0)

    from .store import Store

    store = Store()
    command_str = " ".join(command_parts)
    error_id = store.add_error(command_str, stderr_text, exit_code=returncode)
    matches = store.similar_to(error_id, top_k=3)

    explaining = explain_flag
    if explain_flag:
        _spawn_explain_worker(error_id)

    console_err.print()
    console_err.print(_run_result_panel(error_id, returncode, matches, explaining))
    raise typer.Exit(code=returncode)


@app.command("search")
def search(query: str = typer.Argument(..., help="What went wrong (natural language).")) -> None:
    """Return the most similar past errors + fixes."""
    matches = ErnestClient().search(query, top_k=3)
    if not matches:
        console.print("[yellow]No matching past errors.[/yellow]")
        return
    for m in matches:
        status = "[green]resolved[/green]" if m.resolved else "[yellow]unresolved[/yellow]"
        seen_str = f"  [cyan]seen {m.occurrence_count}x[/cyan]" if m.occurrence_count > 1 else ""
        console.print(f"\n[bold]#{m.id}[/bold]  [dim]{m.score:.0%} match[/dim]  {status}{seen_str}  [dim]{m.date}[/dim]")
        console.print(f"  [dim]cmd:[/dim] {m.failing_command or '?'}")
        console.print(f"  [dim]err:[/dim] {m.error[:200]}")
        if m.fix:
            console.print(f"  [green]fix:[/green] {m.fix}")


@app.command("show")
def show(error_id: int = typer.Argument(..., help="Error id.")) -> None:
    """Show full detail for one error."""
    row = ErnestClient()._store.get(error_id)  # noqa: SLF001 - CLI is store-adjacent
    if row is None:
        console.print(f"[red]No error #{error_id}.[/red]")
        raise typer.Exit(code=1)
    cols = row.keys() if hasattr(row, "keys") else []
    console.print(f"[bold cyan]Error #{row['id']}[/bold cyan]")
    console.print(f"  timestamp:   {row['timestamp']}")
    if "occurrence_count" in cols and row["occurrence_count"] and int(row["occurrence_count"]) > 1:
        first_rel = _relative_time(row["first_seen"]) if "first_seen" in cols else "?"
        last_rel = _relative_time(row["last_seen"]) if "last_seen" in cols else "?"
        console.print(f"  occurrences: [cyan]{row['occurrence_count']}[/cyan] (first seen: {first_rel}, last seen: {last_rel})")
    if "fingerprint" in cols and row["fingerprint"]:
        console.print(f"  fingerprint: {row['fingerprint']}")
    console.print(f"  cwd:         {row['cwd']}")
    console.print(f"  command:     {row['failing_command'] or '-'}")
    console.print(f"  exit code:   {row['exit_code'] if row['exit_code'] is not None else '-'}")
    console.print(f"  git commit:  {row['git_commit'] or '-'}")
    console.print(f"  status:      {row['status']}")
    console.print("\n[bold]Error (redacted):[/bold]")
    console.print((row["error_text_redacted"] or "").strip() or "[dim](none)[/dim]")
    if row["status"] == "resolved" and (row["fix_text"] or "").strip():
        console.print("\n[bold green]Fix:[/bold green]")
        console.print(row["fix_text"])
    if row["ai_explanation"]:
        console.print("\n[bold]AI explanation:[/bold]")
        console.print(row["ai_explanation"])


def _spawn_explain_worker(error_id: int) -> None:
    """Fire-and-forget: run `_explain_worker <id>` fully detached.

    Uses ``sys.executable -m fixerr.cli`` rather than the installed
    ``fixerr`` binary so this works the same in an editable/dev install
    without depending on PATH. ``start_new_session=True`` detaches the child
    from this process's session, so it keeps running (and can finish
    populating the row) even after this command has already exited.
    """
    import subprocess
    import sys

    subprocess.Popen(
        [sys.executable, "-m", "fixerr.cli", "_explain_worker", str(error_id)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


@app.command("_explain_worker", hidden=True)
def explain_worker(error_id: int = typer.Argument(...)) -> None:
    """Internal: generates and stores an AI explanation. Not for direct use.

    Runs fully detached from the command that spawned it — never prints
    anything, never raises. Its only visible effect is ``ai_explanation``
    eventually showing up on the row, whenever generation finishes.
    """
    try:
        from .explain import generate_explanation
        from .store import Store

        store = Store()
        row = store.get(error_id)
        if row is None:
            return
        explanation = generate_explanation(
            row["failing_command"], row["error_text_redacted"] or "", row["fix_text"] or None
        )
        if explanation:
            store.set_ai_explanation(error_id, explanation)
    except Exception:  # noqa: BLE001 - a detached background worker must never crash visibly
        return


@app.command("explain")
def explain(
    error_id: int = typer.Argument(..., help="Error id."),
    regenerate: bool = typer.Option(
        False, "--regenerate", "-r", help="Regenerate even if an explanation is already stored."
    ),
) -> None:
    """Generate (in the background) or show a cached AI explanation.

    Returns immediately — generation happens in a detached process, and the
    result lands in the error's ``ai_explanation`` field whenever it's ready.
    Check back with `fixerr show <id>` or the dashboard.
    """
    from .store import Store

    store = Store()
    row = store.get(error_id)
    if row is None:
        console.print(f"[red]No error #{error_id}.[/red]")
        raise typer.Exit(code=1)

    if row["ai_explanation"] and not regenerate:
        console.print(row["ai_explanation"])
        console.print("[dim]([cyan]--regenerate[/cyan] to ask again)[/dim]")
        return

    _spawn_explain_worker(error_id)
    console.print(
        f"[dim]Generating explanation for #{error_id} in the background — "
        f"check `fixerr show {error_id}` shortly.[/dim]"
    )


# --------------------------------------------------------------- dashboard ---

@app.command("dashboard")
def dashboard() -> None:
    """Interactive TUI over the error/fix store (search, filters, patterns)."""
    from .dashboard import fixerrDashboard

    fixerrDashboard().run()


# ------------------------------------------------------------------- config ---

@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dotted key, e.g. ai.provider"),
    value: str = typer.Argument(..., help="New value, e.g. anthropic"),
) -> None:
    """Set a single config value and persist it to ~/.fixerr/config.toml."""
    try:
        cfg.set_value(key, value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Set[/green] {key} = {value}  [dim]({cfg.config_path()})[/dim]")


@config_app.command("show")
def config_show() -> None:
    """Print the effective config with API-key-bearing values masked."""
    import json

    console.print_json(json.dumps(cfg.masked_config()))


# ------------------------------------------------------------------- doctor ---

def _reachable_backend(backend) -> bool:
    """Best-effort reachability probe that never raises."""
    try:
        backend.embed("ping")
        return True
    except NotImplementedError:
        # Provider can't embed (Anthropic); try generation instead.
        try:
            backend.generate("ping")
            return True
        except Exception:  # noqa: BLE001
            return False
    except Exception:  # noqa: BLE001
        return False


def _mark(ok: bool) -> str:
    return "[green]✓[/green]" if ok else "[red]✗[/red]"


@app.command("doctor")
def doctor() -> None:
    """Check the configured AI provider, models, and credentials."""
    ai = cfg.load_config().get("ai", {})
    provider = str(ai.get("provider", "ollama")).lower()
    section = ai.get(provider, {}) if isinstance(ai.get(provider), dict) else {}

    console.print("[bold]fixerr doctor[/bold]")

    known = {"ollama", "openai", "openai-compatible", "anthropic", "gemini"}
    console.print(f"├── provider: {provider} {_mark(provider in known)}")

    # Embed fallback (only meaningful when the provider can't embed itself)
    if provider == "anthropic":
        fallback = str(ai.get("embed_fallback", "ollama"))
        try:
            reachable = _reachable_backend(get_embed_backend())
        except Exception:  # noqa: BLE001
            reachable = False
        state = "reachable" if reachable else "unreachable"
        console.print(f"├── embed fallback: {fallback} {_mark(reachable)} ({state})")

    # API key env var (skip for ollama, which needs none)
    key_env = section.get("api_key_env")
    if key_env and str(key_env).lower() != "none":
        is_set = bool(os.environ.get(key_env))
        console.print(f"├── {key_env}: {'set' if is_set else 'missing'} {_mark(is_set)}")

    # Generation model reachability
    gen_model = section.get("gen_model", "?")
    try:
        gen_ok = _reachable_backend(get_backend())
    except Exception:  # noqa: BLE001
        gen_ok = False
    gen_state = "reachable" if gen_ok else "unreachable"
    console.print(f"├── gen_model: {gen_model} {_mark(gen_ok)} ({gen_state})")

    # Embedding model reachability (routes through embed_fallback if needed)
    embed_provider = "ollama" if provider == "anthropic" else provider
    embed_section = ai.get(embed_provider, {}) if isinstance(ai.get(embed_provider), dict) else {}
    embed_model = embed_section.get("embed_model", "?")
    try:
        embed_ok = _reachable_backend(get_embed_backend())
    except Exception:  # noqa: BLE001
        embed_ok = False
    console.print(f"└── embed_model: {embed_model} via {embed_provider} {_mark(embed_ok)}")


def main() -> None:  # console_scripts entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
