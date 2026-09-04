"""``fixerr dashboard`` — an interactive Textual TUI over the error/fix store.

Real SQLite-backed data (via :mod:`fixerr.store` directly — one shared
``Store`` instance, no duplicate DB connections or Ollama calls),
selection-driven detail view, tag/status filtering, search (semantic when
embeddings are reachable, degrading to a substring filter otherwise),
resolve/dismiss/copy-fix mutations, a semantically-similar-errors panel, and a
Patterns tab that clusters captured errors via a small pure-Python k-means
(no numpy dependency) over their stored embeddings.
"""

from __future__ import annotations

import asyncio
import math
import random
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

from .ai import AIUnavailableError, get_backend
from .config import load_config
from .store import STATUS_OPEN, STATUS_RESOLVED, STATUS_WONT_FIX, Match, Store


async def run_in_daemon_thread(func, *args):
    """Like ``asyncio.to_thread``, but on a throwaway ``daemon=True`` thread.

    ``asyncio.to_thread`` runs on the default ``ThreadPoolExecutor``, whose
    worker threads are *not* daemon threads — and ``concurrent.futures``
    registers an atexit hook that joins all pending work before the
    interpreter can exit. A slow/hanging Ollama call still in flight (up to
    the configured ``gen_timeout``) would silently block 'q'/Ctrl-C from
    actually terminating the process. A plain daemon thread is invisible to
    that hook: if it's still running when the app exits, Python just drops it.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def deliver(value, is_exception: bool) -> None:
        try:
            if is_exception:
                future.set_exception(value)
            else:
                future.set_result(value)
        except asyncio.InvalidStateError:
            pass  # awaiter already gave up (e.g. cancelled) — nothing to deliver to

    def runner() -> None:
        try:
            result = func(*args)
        except Exception as exc:  # noqa: BLE001 - propagated to the awaiter below
            error = exc
        else:
            error = None
        try:
            # The app (and its event loop) may have already shut down by the
            # time this daemon thread finishes — that's expected, not a bug.
            if error is not None:
                loop.call_soon_threadsafe(deliver, error, True)
            else:
                loop.call_soon_threadsafe(deliver, result, False)
        except RuntimeError:
            pass  # event loop is closed — the app already exited

    threading.Thread(target=runner, daemon=True).start()
    return await future


def copy_to_system_clipboard(text: str) -> bool:
    """Best-effort clipboard copy via a native OS command.

    Textual's own ``App.copy_to_clipboard`` writes an OSC 52 terminal escape
    sequence, which many terminals — notably macOS's default Terminal.app —
    don't support, so it silently does nothing there. A native command-line
    tool is far more reliable for the common case (a local terminal session).
    Returns True if a copy command actually ran successfully.
    """
    import platform
    import subprocess

    system = platform.system()
    if system == "Darwin":
        candidates = [["pbcopy"]]
    elif system == "Linux":
        candidates = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    elif system == "Windows":
        candidates = [["clip"]]
    else:
        candidates = []

    for cmd in candidates:
        try:
            subprocess.run(
                cmd,
                input=text.encode(),
                check=True,
                timeout=3,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


_STATUS_BADGE = {
    "new": "[bold cyan]● new[/bold cyan]",
    "open": "[bold yellow]● open[/bold yellow]",
    STATUS_RESOLVED: "[bold green]✓ resolved[/bold green]",
    STATUS_WONT_FIX: "[dim]✗ wont-fix[/dim]",
}

_TAG_CHIPS = ["all", "docker", "python", "port", "venv", "network"]

_TAB_ALL = "all"
_TAB_UNRESOLVED = "unresolved"
_TAB_RESOLVED = "resolved"
_TAB_PATTERNS = "patterns"


# ---------------------------------------------------------------- tag rules ---

def derive_tags(match: Match) -> List[str]:
    """Auto-detect tags from error text + command — no DB column, computed at display time."""
    cmd = (match.failing_command or "").lower()
    err = (match.error or "").lower()
    tags: List[str] = []
    if "docker" in cmd or "docker" in err:
        tags.append("docker")
    if "port" in err or "bind" in err or "already in use" in err:
        tags.append("port")
    if "modulenotfounderror" in err or "importerror" in err or "pip" in cmd or "pip" in err:
        tags.append("python")
    if "venv" in err or "activate" in err or "no module named" in err:
        tags.append("venv")
    if "network" in err and "docker" in err:
        tags.append("network")
    return tags


def _badge_for(match: Match) -> str:
    if match.status != STATUS_OPEN:
        return _STATUS_BADGE.get(match.status, match.status)
    # "new" vs "open" is an age-based display label over the same underlying
    # open state — captured within the last 24h reads as "new".
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(match.timestamp)
        if age.total_seconds() < 24 * 3600:
            return _STATUS_BADGE["new"]
    except (ValueError, TypeError):
        pass
    return _STATUS_BADGE["open"]


def _format_iso_relative(iso_ts: Optional[str]) -> str:
    if not iso_ts:
        return "?"
    try:
        ts = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return (iso_ts or "")[:10]
    now = datetime.now(timezone.utc) if ts.tzinfo else datetime.now()
    secs = (now - ts).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def _relative_date(match: Match) -> str:
    return _format_iso_relative(match.last_seen or match.timestamp)


# ------------------------------------------------------------------- widgets ---


class StatsBar(Static):
    """Open / resolved / wont-fix counts, AI coverage %, most frequent tag."""

    def render_stats(self, errors: List[Match]) -> str:
        if not errors:
            return "[dim]No errors captured yet.[/dim]"
        open_n = sum(1 for e in errors if e.status == STATUS_OPEN)
        resolved_n = sum(1 for e in errors if e.status == STATUS_RESOLVED)
        wontfix_n = sum(1 for e in errors if e.status == STATUS_WONT_FIX)
        with_ai = sum(1 for e in errors if e.ai_explanation)
        coverage = (with_ai / len(errors) * 100) if errors else 0.0
        top_tag = self._top_tag(errors)
        return (
            f"[bold]{open_n}[/bold] open    "
            f"[green]{resolved_n}[/green] resolved    "
            f"[dim]{wontfix_n}[/dim] wont-fix    "
            f"AI coverage [bold]{coverage:.0f}%[/bold]    "
            f"most frequent: [bold cyan]{top_tag}[/bold cyan]"
        )

    @staticmethod
    def _top_tag(errors: List[Match]) -> str:
        counts: dict[str, int] = {}
        for e in errors:
            for t in derive_tags(e):
                counts[t] = counts.get(t, 0) + 1
        if not counts:
            return "—"
        return max(counts, key=counts.get)


class TabRow(Static):
    """All | Unresolved | Resolved | Patterns, with live counts. Click or 1-4/p to switch."""

    active: reactive[str] = reactive(_TAB_ALL)

    def render_row(self, errors: List[Match]) -> str:
        counts = {
            _TAB_ALL: len(errors),
            _TAB_UNRESOLVED: sum(1 for e in errors if e.status == STATUS_OPEN),
            _TAB_RESOLVED: sum(1 for e in errors if e.status == STATUS_RESOLVED),
        }
        labels = [
            (_TAB_ALL, f"All ({counts[_TAB_ALL]})"),
            (_TAB_UNRESOLVED, f"Unresolved ({counts[_TAB_UNRESOLVED]})"),
            (_TAB_RESOLVED, f"Resolved ({counts[_TAB_RESOLVED]})"),
            (_TAB_PATTERNS, "Patterns"),
        ]
        parts = []
        for key, label in labels:
            style = "reverse bold" if key == self.active else "dim"
            parts.append(f"[{style}] {label} [/{style}]")
        return "  ".join(parts)


class _Chip(Static):
    """A single clickable filter chip. Reports clicks via a plain callback
    (not Textual's Message/bubble system — avoids fragile handler-name magic
    for what is otherwise a single, simple parent-child relationship)."""

    can_focus = True

    def __init__(self, tag: str, on_click_cb) -> None:
        super().__init__(tag)
        self.tag_name = tag
        self._on_click_cb = on_click_cb

    def set_active(self, active: bool) -> None:
        self.set_class(active, "-active")

    def on_click(self) -> None:
        self._on_click_cb(self.tag_name)


class FilterChips(Horizontal):
    """Row of clickable tag filter chips: all | docker | python | port | venv | network."""

    can_focus = True
    active: reactive[str] = reactive("all")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.on_select = None  # callback(tag: str), set by the App after mount

    def compose(self) -> ComposeResult:
        for chip in _TAG_CHIPS:
            yield _Chip(chip, self._handle_click)

    def _handle_click(self, tag: str) -> None:
        self.active = tag
        if self.on_select:
            self.on_select(tag)

    def on_mount(self) -> None:
        self._sync_active()

    def watch_active(self, active: str) -> None:
        self._sync_active()

    def _sync_active(self) -> None:
        for chip in self.query(_Chip):
            chip.set_active(chip.tag_name == self.active)


class ErrorListPanel(Vertical):
    """Left panel: search input, filter chips, scrollable error rows."""

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search errors…", id="search-input")
        yield FilterChips(id="filter-chips")
        yield ListView(id="error-list")

    @staticmethod
    def row_for(match: Match) -> ListItem:
        badge = _badge_for(match)
        cmd = match.failing_command or "?"
        cmd = cmd if len(cmd) <= 30 else cmd[:27] + "…"
        first_line = (match.error or "").splitlines()[0] if match.error else ""
        if len(first_line) > 42:
            first_line = first_line[:39] + "…"
        count_badge = f" [cyan]×{match.occurrence_count}[/cyan]" if match.occurrence_count > 1 else ""
        tags = " ".join(f"#{t}" for t in derive_tags(match))
        text = f"{cmd}{count_badge}\n{badge}  [dim]{_relative_date(match)}[/dim]\n[dim]{first_line}[/dim]"
        if tags:
            text += f"\n[cyan]{tags}[/cyan]"
        item = ListItem(Static(text))
        item.id = f"error-{match.id}"
        return item


class _SimilarRow(Static):
    """A single clickable 'similar error' row (id, similarity bar, snippet)."""

    can_focus = True

    def __init__(self, match_id: int, label: str, on_click_cb) -> None:
        super().__init__(label)
        self.match_id = match_id
        self._on_click_cb = on_click_cb

    def on_click(self) -> None:
        self._on_click_cb(self.match_id)


class DetailPanel(VerticalScroll):
    """Right panel: header/actions, metadata grid, error output, fix, AI explanation, similar errors."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.on_similar_click = None  # callback(error_id: int), set by the App after mount

    def compose(self) -> ComposeResult:
        yield Static(id="detail-title")
        yield Static(id="detail-actions")
        yield Static(id="detail-metadata")
        yield Label("[bold]Error output[/bold]")
        yield Static(id="detail-error", classes="code-block")
        yield Label("[bold]Fix applied[/bold]")
        yield Static(id="detail-fix", classes="code-block")
        yield Label("[bold]AI explanation[/bold]")
        yield Static(id="detail-ai", classes="code-block")
        yield Label("[bold]Semantically similar errors[/bold]", id="similar-label")
        yield Vertical(id="detail-similar")

    def show_empty(self) -> None:
        self.query_one("#detail-title", Static).update("[dim]No error selected.[/dim]")
        for wid in ("detail-actions", "detail-metadata", "detail-error", "detail-fix", "detail-ai"):
            self.query_one(f"#{wid}", Static).update("")
        self.query_one("#similar-label", Label).display = False
        self.query_one("#detail-similar", Vertical).display = False

    async def show(self, match: Match, similar: Optional[List[Match]], explaining: bool = False) -> None:
        self.query_one("#detail-title", Static).update(f"[bold]#{match.id}  {match.failing_command or '?'}[/bold]")
        if match.status == STATUS_OPEN:
            resolve_label = "[reverse] r mark resolved [/reverse]"
            dismiss_label = "[reverse] d dismiss [/reverse]"
            reopen_label = "[dim]already open[/dim]"
        elif match.status == STATUS_RESOLVED:
            resolve_label = "[dim]already resolved[/dim]"
            dismiss_label = "[reverse] d dismiss [/reverse]"
            reopen_label = "[reverse] u reopen [/reverse]"
        else:  # wont-fix
            resolve_label = "[reverse] r mark resolved [/reverse]"
            dismiss_label = "[dim]already dismissed[/dim]"
            reopen_label = "[reverse] u reopen [/reverse]"
        self.query_one("#detail-actions", Static).update(
            f"[reverse] c copy fix [/reverse]  {resolve_label}  {dismiss_label}  "
            f"{reopen_label}  [reverse] e explain [/reverse]"
        )
        meta = (
            f"command:     {match.failing_command or '-'}\n"
            f"exit code:   {match.exit_code if match.exit_code is not None else '—'}\n"
            f"directory:   {match.cwd or '—'}\n"
            f"git commit:  {match.git_commit or '—'}\n"
            f"captured:    {match.timestamp}\n"
            f"resolved:    {match.resolved_at or '—'}"
        )
        if match.occurrence_count > 1:
            first_rel = _format_iso_relative(match.first_seen)
            last_rel = _format_iso_relative(match.last_seen)
            meta += f"\noccurrences: {match.occurrence_count} (first: {first_rel}, last: {last_rel})"
        if match.fingerprint:
            meta += f"\nfingerprint: {match.fingerprint}"
        self.query_one("#detail-metadata", Static).update(meta)
        self.query_one("#detail-error", Static).update(match.error or "[dim](no error text)[/dim]")
        self.query_one("#detail-fix", Static).update(
            match.fix or "[dim]Not resolved yet — press 'r' to mark resolved.[/dim]"
        )
        if explaining:
            # A persistent in-panel indicator, not just a toast — local models
            # can genuinely take tens of seconds, and the toast notification
            # disappears long before generation finishes, which otherwise
            # reads as "nothing is happening" rather than "still working".
            self.query_one("#detail-ai", Static).update(
                "[dim]Asking the model… this can take a while for local models.[/dim]"
            )
        elif match.ai_explanation:
            model = _current_gen_model()
            self.query_one("#detail-ai", Static).update(f"[dim](AI-inferred · {model})[/dim]\n{match.ai_explanation}")
        else:
            self.query_one("#detail-ai", Static).update(
                "[dim]Press 'e' to ask the model to explain this error.[/dim]"
            )
        await self._show_similar(similar)

    async def _show_similar(self, similar: Optional[List[Match]]) -> None:
        label = self.query_one("#similar-label", Label)
        container = self.query_one("#detail-similar", Vertical)
        if similar is None:
            # Embeddings unreachable — hide the section entirely, no error shown.
            label.display = False
            container.display = False
            return
        label.display = True
        container.display = True
        await container.remove_children()
        if not similar:
            await container.mount(Static("[dim]No similar errors found.[/dim]"))
            return
        rows = []
        for m in similar:
            pct = round(m.score * 100)
            filled = max(0, min(20, round(pct / 5)))
            bar = "█" * filled + "░" * (20 - filled)
            snippet = (m.error or "").splitlines()[0] if m.error else ""
            if len(snippet) > 46:
                snippet = snippet[:43] + "…"
            row_text = f"#{m.id}  {pct:>3}%  {bar}  {snippet}"
            rows.append(_SimilarRow(m.id, row_text, self._on_similar_row_click))
        await container.mount_all(rows)

    def _on_similar_row_click(self, error_id: int) -> None:
        if self.on_similar_click:
            self.on_similar_click(error_id)


def _current_gen_model() -> str:
    ai = load_config().get("ai", {})
    provider = str(ai.get("provider", "ollama"))
    section = ai.get(provider, {}) if isinstance(ai.get(provider), dict) else {}
    return str(section.get("gen_model", provider))


# ------------------------------------------------------------------ k-means ---
# Pure-Python k-means (no numpy) — the error/embedding counts this runs over
# (tens to low hundreds of local errors, not a production ML workload) don't
# justify the dependency.

def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def kmeans(vectors: List[List[float]], k: int, iterations: int = 25, seed: int = 42) -> List[int]:
    """Cluster ``vectors`` (already L2-normalized) into ``k`` clusters.

    Returns a cluster index per input vector. Normalizing first makes
    Euclidean distance monotonic with cosine similarity, matching how the
    rest of fixerr compares embeddings.
    """
    n = len(vectors)
    k = max(1, min(k, n))
    rng = random.Random(seed)
    centroids = [list(v) for v in rng.sample(vectors, k)]
    assignments = [0] * n

    for _ in range(iterations):
        changed = False
        for i, v in enumerate(vectors):
            dists = [_euclidean(v, c) for c in centroids]
            best = dists.index(min(dists))
            if assignments[i] != best:
                changed = True
            assignments[i] = best

        new_centroids = []
        for ci in range(k):
            members = [vectors[i] for i in range(n) if assignments[i] == ci]
            if not members:
                new_centroids.append(centroids[ci])  # keep stale centroid; no member to average
                continue
            dim = len(members[0])
            new_centroids.append([sum(m[d] for m in members) / len(members) for d in range(dim)])
        centroids = new_centroids

        if not changed:
            break

    return assignments


@dataclass
class ClusterSummary:
    label: str
    count: int
    examples: List[str]
    resolution_rate: float


def _command_prefix(command: Optional[str]) -> str:
    cmd = (command or "").strip()
    return cmd.split()[0] if cmd else "?"


def compute_clusters(pairs: List[Tuple[Match, List[float]]], k: int = 5) -> List[ClusterSummary]:
    """Cluster (Match, embedding) pairs and summarize each cluster for display."""
    if len(pairs) < 2:
        return []
    matches = [m for m, _ in pairs]
    vectors = [_normalize(vec) for _, vec in pairs]
    assignments = kmeans(vectors, k)

    grouped: Dict[int, List[Match]] = defaultdict(list)
    for idx, cluster_id in enumerate(assignments):
        grouped[cluster_id].append(matches[idx])

    summaries = []
    for members in grouped.values():
        prefixes = Counter(_command_prefix(m.failing_command) for m in members)
        label = prefixes.most_common(1)[0][0]
        examples = []
        for m in members[:2]:
            line = (m.error or "").splitlines()[0] if m.error else ""
            examples.append(line[:70] + ("…" if len(line) > 70 else ""))
        resolved_n = sum(1 for m in members if m.status == STATUS_RESOLVED)
        summaries.append(
            ClusterSummary(
                label=label,
                count=len(members),
                examples=examples,
                resolution_rate=(resolved_n / len(members) * 100) if members else 0.0,
            )
        )
    summaries.sort(key=lambda c: c.count, reverse=True)
    return summaries


class PatternsView(Static):
    """Full-width Patterns tab: k-means clusters over stored embeddings."""

    def show_need_more(self, current: int) -> None:
        self.update(f"[dim]Need 20+ errors for pattern analysis — currently {current}.[/dim]")

    def show_clusters(self, clusters: List[ClusterSummary]) -> None:
        if not clusters:
            self.update(
                "[dim]No clusterable errors yet — embeddings are missing for every "
                "captured error (was the embed backend down when they were captured?).[/dim]"
            )
            return
        lines = []
        for i, c in enumerate(clusters, 1):
            lines.append(
                f"[bold cyan]Cluster {i}: {c.label}[/bold cyan]  "
                f"[dim]{c.count} error{'s' if c.count != 1 else ''} · "
                f"{c.resolution_rate:.0f}% resolved[/dim]"
            )
            for example in c.examples:
                lines.append(f"    [dim]{example}[/dim]")
            lines.append("")
        self.update("\n".join(lines).rstrip())


class MainArea(Horizontal):
    def compose(self) -> ComposeResult:
        yield ErrorListPanel(id="left-panel")
        yield DetailPanel(id="right-panel")


# ---------------------------------------------------------------------- app ---


class fixerrDashboard(App):
    """fixerr's interactive TUI dashboard."""

    CSS = """
    /* ------------------------------------------------------------------
       fixerr terminal theme
       The layout is intentionally compact: the error queue gets ~1/3 of
       the screen while the selected error gets the remaining space.
    ------------------------------------------------------------------ */
    Screen {
        background: #0b0f14;
        color: #d7dee8;
    }

    Header {
        height: 3;
        background: #101722;
        color: #7dd3fc;
        border-bottom: solid #263241;
    }

    Footer {
        height: 1;
        background: #101722;
        color: #94a3b8;
        border-top: solid #263241;
    }

    #content {
        height: 1fr;
        width: 1fr;
    }

    /* ------------------------------- top information strip ------------ */
    StatsBar {
        height: 2;
        padding: 0 2;
        background: #111923;
        color: #cbd5e1;
        border-bottom: solid #263241;
        content-align: left middle;
    }

    TabRow {
        height: 2;
        padding: 0 2;
        background: #0e141d;
        color: #94a3b8;
        border-bottom: solid #263241;
        content-align: left middle;
    }

    /* ----------------------------------- left error queue ------------- */
    #left-panel {
        width: 34%;
        min-width: 34;
        height: 1fr;
        padding: 1 1;
        background: #0d131b;
        border-right: solid #334155;
    }

    #search-input {
        height: 3;
        margin-bottom: 1;
        background: #111923;
        border: round #334155;
        color: #e2e8f0;
    }

    #search-input:focus {
        border: round #38bdf8;
    }

    #filter-chips {
        height: 2;
        margin-bottom: 1;
        overflow-x: hidden;
        content-align: left middle;
    }

    /* Filter labels: plain text, no empty-looking boxes. */
    _Chip {
        width: auto;
        min-width: 4;
        height: 1;
        margin-right: 2;
        padding: 0;
        color: #94a3b8;
        background: transparent;
        border: none;
        content-align: center middle;
    }

    _Chip.-active {
        padding: 0 1;
        color: #e0f2fe;
        background: #075985;
        border: none;
        text-style: bold;
    }

    #error-list {
        height: 1fr;
        background: transparent;
        border: none;
        scrollbar-color: #334155;
        scrollbar-color-hover: #475569;
    }

    #error-list > ListItem {
        min-height: 4;
        margin-bottom: 1;
        padding: 1 1;
        background: #111923;
        border: round #1e293b;
    }

    #error-list > ListItem:hover {
        background: #172231;
        border: round #3b82f6;
    }

    /* Strong visual indication for the currently selected/clicked error. */
    #error-list > ListItem.-highlight {
        background: #075985;
        border: round #38bdf8;
        color: #f8fafc;
        text-style: bold;
    }

    #error-list > ListItem.-highlight Static {
        color: #f8fafc;
    }

    /* -------------------------------- selected error ------------------ */
    #right-panel {
        width: 66%;
        height: 1fr;
        padding: 1 2;
        background: #0b0f14;
        scrollbar-color: #334155;
    }

    #detail-title {
        height: 2;
        padding: 0 1;
        color: #f1f5f9;
        background: #111923;
        border-bottom: solid #38bdf8;
        content-align: left middle;
    }

    #detail-actions {
        height: 2;
        margin: 1 0;
        padding: 0 1;
        color: #cbd5e1;
        background: #0f1720;
        content-align: left middle;
    }

    #detail-metadata {
        padding: 1 1;
        margin-bottom: 1;
        background: #0f1720;
        border: round #263241;
        color: #aebdca;
    }

    Label {
        color: #dbeafe;
        text-style: bold;
        margin-top: 0;
        margin-bottom: 0;
    }

    .code-block {
        min-height: 3;
        background: #18222d;
        border: round #334155;
        padding: 1 2;
        margin-top: 0;
        margin-bottom: 1;
        color: #cbd5e1;
    }

    #detail-error {
        border: round #f59e0b;
    }

    #detail-fix {
        border: round #22c55e;
    }

    #detail-ai {
        border: round #38bdf8;
    }

    #similar-label {
        margin-top: 1;
        color: #c4b5fd;
    }

    #detail-similar {
        margin-top: 0;
        padding: 0 1;
        background: #0f1720;
        border: round #263241;
    }

    _SimilarRow {
        height: 2;
        margin-bottom: 0;
        padding: 0 1;
        color: #cbd5e1;
    }

    _SimilarRow:hover {
        background: #172231;
        color: #f8fafc;
        text-style: bold;
    }

    #patterns-view {
        height: 1fr;
        padding: 2 3;
        background: #0d131b;
        border: round #334155;
        color: #cbd5e1;
    }

    #onboarding {
        content-align: center middle;
        height: 100%;
        padding: 2;
        color: #94a3b8;
    }
    """

    BINDINGS = [
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("r", "mark_resolved", "Resolve"),
        ("d", "dismiss", "Dismiss"),
        ("u", "reopen", "Reopen"),
        ("c", "copy_fix", "Copy fix"),
        ("e", "explain", "Explain"),
        ("s", "focus_search", "Search"),
        ("/", "focus_filters", "Filters"),
        ("p", "show_patterns", "Patterns"),
        ("1", "tab_all", "All"),
        ("2", "tab_unresolved", "Unresolved"),
        ("3", "tab_resolved", "Resolved"),
        ("4", "tab_patterns", "Patterns"),
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, store: Optional[Store] = None) -> None:
        super().__init__()
        # One shared Store for the whole app — no duplicate DB connections.
        self._store = store or Store()
        self._all_errors: List[Match] = []
        self._visible: List[Match] = []
        self._selected_id: Optional[int] = None
        self._active_tab = _TAB_ALL
        self._active_tag = "all"
        self._search_query = ""
        self._embed_offline = False
        self._patterns_cache: Optional[List[ClusterSummary]] = None
        self._search_timer = None
        self._explaining_ids: set = set()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield StatsBar()
        yield TabRow()
        with Vertical(id="content"):
            yield MainArea()
            yield PatternsView(id="patterns-view")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "fixerr"
        self.query_one("#patterns-view", PatternsView).display = False
        self.query_one(FilterChips).on_select = self._on_tag_selected
        self.query_one(DetailPanel).on_similar_click = self._on_similar_click
        self.reload_data()
        # Input auto-focuses on mount and would otherwise swallow every
        # keystroke (including 1/2/3/4/p/r/d/c) as literal text. Default
        # focus to the list so single-key bindings work immediately; 's'
        # moves focus to search explicitly, and escape/enter return it here.
        self.query_one("#error-list", ListView).focus()

    def on_key(self, event) -> None:
        if event.key in ("escape", "enter") and self.focused is self.query_one("#search-input", Input):
            self.query_one("#error-list", ListView).focus()

    def _on_tag_selected(self, tag: str) -> None:
        self._active_tag = tag
        self.run_worker(self._apply_filters(), exclusive=True)

    def _on_similar_click(self, error_id: int) -> None:
        # Clicking a similar-error row navigates to it in the left panel —
        # reset filters so it's guaranteed visible regardless of current tab/
        # tag/search state, then select it.
        self._active_tab = _TAB_ALL
        self._active_tag = "all"
        self._search_query = ""
        self.query_one("#search-input", Input).value = ""
        self.query_one(TabRow).active = _TAB_ALL
        self.query_one(FilterChips).active = "all"
        main = self.query_one(MainArea)
        patterns = self.query_one("#patterns-view", PatternsView)
        main.display = True
        patterns.display = False
        self._selected_id = error_id
        self.run_worker(self._apply_filters(), exclusive=True)

    # -- data loading ---------------------------------------------------------
    def reload_data(self) -> None:
        self._all_errors = self._store.list_errors()
        self._refresh_local_header()
        # Reachability probes hit the network (Ollama/OpenAI/etc.) and must
        # never block startup or a reload — a slow/hanging backend used to
        # freeze the whole UI here for up to the configured generate timeout
        # (which can legitimately be minutes now that `explain`/`run`'s
        # detached callers want a generous one). Backgrounded via a thread so
        # the header just shows "checking..." until it resolves.
        self.run_worker(self._reload_provider_status(), exclusive=True, group="provider-status")
        self.run_worker(self._apply_filters(), exclusive=True)

    def _refresh_local_header(self) -> None:
        """The parts of the header that need no network — always instant."""
        n = len(self._all_errors)
        self.sub_title = f"provider: checking… · {n} error{'s' if n != 1 else ''}"
        self.query_one(StatsBar).update(self.query_one(StatsBar).render_stats(self._all_errors))
        self.query_one(TabRow).update(self.query_one(TabRow).render_row(self._all_errors))

    async def _reload_provider_status(self) -> None:
        provider, model = await run_in_daemon_thread(self._provider_status)
        self._embed_offline = await run_in_daemon_thread(self._probe_embed_offline)
        n = len(self._all_errors)
        self.sub_title = f"provider: {provider} · {model} · {n} error{'s' if n != 1 else ''}"

    @staticmethod
    def _probe_embed_offline() -> bool:
        from .ai import get_embed_backend

        try:
            get_embed_backend().embed("ping")
        except Exception:  # noqa: BLE001
            return True
        return False

    @staticmethod
    def _provider_status() -> tuple[str, str]:
        ai = load_config().get("ai", {})
        provider = str(ai.get("provider", "ollama"))
        section = ai.get(provider, {}) if isinstance(ai.get(provider), dict) else {}
        model = str(section.get("gen_model", "?"))
        try:
            get_backend().generate("ping")
        except AIUnavailableError:
            return "offline", model
        except Exception:  # noqa: BLE001
            return "offline", model
        return provider, model

    # -- filtering --------------------------------------------------------
    async def _apply_filters(self) -> None:
        errors = self._all_errors

        if self._active_tab == _TAB_UNRESOLVED:
            errors = [e for e in errors if e.status == STATUS_OPEN]
        elif self._active_tab == _TAB_RESOLVED:
            errors = [e for e in errors if e.status == STATUS_RESOLVED]

        if self._active_tag != "all":
            errors = [e for e in errors if self._active_tag in derive_tags(e)]

        if self._search_query.strip():
            errors = self._search(self._search_query.strip(), errors)

        self._visible = errors
        await self._render_list()

    def _substring_filter(self, query: str, pool: List[Match]) -> List[Match]:
        ql = query.lower()
        return [
            e for e in pool
            if ql in (e.failing_command or "").lower() or ql in (e.error or "").lower()
        ]

    def _search(self, query: str, pool: List[Match]) -> List[Match]:
        if len(query) <= 2:
            return self._substring_filter(query, pool)
        if self._embed_offline:
            # No embeddings reachable — degrade to a substring filter (the
            # dashboard's stand-in for "SQL LIKE"; the pool is already loaded
            # in memory, so this needs no extra query).
            return self._substring_filter(query, pool)
        # Embeddings reachable: rank the pool by semantic similarity. This is
        # a *reorder*, not a hard include/exclude filter — standard semantic
        # search UX (store.search() itself applies no relevance threshold).
        pool_ids = {e.id for e in pool}
        ranked = self._store.search(query, top_k=len(pool))
        by_id = {e.id: e for e in pool}
        return [by_id[m.id] for m in ranked if m.id in pool_ids]

    async def _render_list(self) -> None:
        list_view = self.query_one("#error-list", ListView)
        await list_view.clear()
        left = self.query_one("#left-panel", ErrorListPanel)
        onboarding = self.query(".onboarding")
        if not self._all_errors:
            if not onboarding:
                await left.mount(
                    Static(
                        "No errors captured yet.\n\n"
                        "After a command fails, run:\n"
                        "  the-command 2>&1 | fixerr capture -c \"the-command\"",
                        id="onboarding",
                        classes="onboarding",
                    )
                )
            return
        for widget in onboarding:
            await widget.remove()
        await list_view.extend(ErrorListPanel.row_for(match) for match in self._visible)
        if self._visible:
            ids = [e.id for e in self._visible]
            if self._selected_id not in ids:
                self._selected_id = ids[0]
            # `clear()` resets index to None; give it a concrete starting
            # index so j/k (cursor_down/up) advance from a known position
            # instead of re-highlighting the same first row on first press.
            list_view.index = ids.index(self._selected_id)
            await self._show_selected()
        else:
            self._selected_id = None
            self.query_one("#right-panel", DetailPanel).show_empty()

    async def _show_selected(self) -> None:
        match = next((e for e in self._visible if e.id == self._selected_id), None)
        if match is not None:
            similar = self._compute_similar(match)
            explaining = match.id in self._explaining_ids
            await self.query_one("#right-panel", DetailPanel).show(match, similar, explaining)

    def _compute_similar(self, match: Match) -> Optional[List[Match]]:
        if self._embed_offline:
            return None
        return self._store.similar_to(match.id, top_k=3)

    # -- selection ----------------------------------------------------------
    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is None or event.item.id is None:
            return
        self._selected_id = int(event.item.id.removeprefix("error-"))
        self.run_worker(self._show_selected(), exclusive=True, group="detail")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search-input":
            return
        self._search_query = event.value
        if self._search_timer is not None:
            self._search_timer.stop()
        # Debounce 300ms so we don't re-run search on every keystroke.
        self._search_timer = self.set_timer(0.3, self._debounced_search)

    def _debounced_search(self) -> None:
        self.run_worker(self._apply_filters(), exclusive=True)

    # -- tab / tag switching -------------------------------------------------
    def _set_tab(self, tab: str) -> None:
        self._active_tab = tab
        self.query_one(TabRow).active = tab
        self.query_one(TabRow).update(self.query_one(TabRow).render_row(self._all_errors))
        patterns = self.query_one("#patterns-view", PatternsView)
        main = self.query_one(MainArea)
        if tab == _TAB_PATTERNS:
            main.display = False
            patterns.display = True
            n = len(self._all_errors)
            if n < 20:
                patterns.show_need_more(n)
            else:
                if self._patterns_cache is None:
                    # Computed once per session, not on every tab switch —
                    # k-means over embeddings isn't free, and the spec is
                    # explicit that switching tabs shouldn't re-run it.
                    pairs = self._store.list_with_embeddings()
                    self._patterns_cache = compute_clusters(pairs, k=5)
                patterns.show_clusters(self._patterns_cache)
        else:
            main.display = True
            patterns.display = False
            self.run_worker(self._apply_filters(), exclusive=True)

    def action_tab_all(self) -> None:
        self._set_tab(_TAB_ALL)

    def action_tab_unresolved(self) -> None:
        self._set_tab(_TAB_UNRESOLVED)

    def action_tab_resolved(self) -> None:
        self._set_tab(_TAB_RESOLVED)

    def action_tab_patterns(self) -> None:
        self._set_tab(_TAB_PATTERNS)

    def action_show_patterns(self) -> None:
        self._set_tab(_TAB_PATTERNS)

    # -- navigation -----------------------------------------------------------
    def action_cursor_down(self) -> None:
        self.query_one("#error-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#error-list", ListView).action_cursor_up()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_focus_filters(self) -> None:
        self.query_one("#filter-chips", FilterChips).focus()

    # -- mutation actions -----------------------------------------------------
    def _selected_match(self) -> Optional[Match]:
        if self._selected_id is None:
            return None
        return next((e for e in self._all_errors if e.id == self._selected_id), None)

    def action_mark_resolved(self) -> None:
        match = self._selected_match()
        if match is None or match.status == STATUS_RESOLVED:
            return
        # Dashboard's 'r' is a quick "mark done" toggle — it keeps whatever
        # fix text already exists (often none). `fixerr resolve <id> "..."`
        # remains the way to attach detailed fix text.
        self._store.set_fix(match.id, match.fix)
        self.notify(f"Error #{match.id} marked resolved.")
        self.reload_data()

    def action_dismiss(self) -> None:
        match = self._selected_match()
        if match is None or match.status == STATUS_WONT_FIX:
            return
        self._store.dismiss(match.id)
        self.notify(f"Error #{match.id} marked won't-fix.")
        self.reload_data()

    def action_reopen(self) -> None:
        match = self._selected_match()
        if match is None or match.status == STATUS_OPEN:
            return
        self._store.reopen(match.id)
        self.notify(f"Error #{match.id} reopened.")
        self.reload_data()

    def action_copy_fix(self) -> None:
        match = self._selected_match()
        if match is None:
            return
        if not match.fix:
            self.notify("No fix text to copy yet.", severity="warning")
            return
        # Try a native clipboard tool first — much more reliable than
        # Textual's OSC 52 escape sequence, which macOS's default
        # Terminal.app (and several others) don't support and silently
        # no-ops on. Fall back to OSC 52 for terminals/SSH sessions where
        # it's the only thing that can reach the *local* clipboard.
        if copy_to_system_clipboard(match.fix):
            self.notify(f"Copied fix for #{match.id} to clipboard.")
        else:
            self.copy_to_clipboard(match.fix)
            self.notify(
                f"Sent fix for #{match.id} to the terminal's clipboard sync "
                f"(no native clipboard tool found — this only works if your "
                f"terminal supports OSC 52).",
                severity="warning",
            )

    def action_explain(self) -> None:
        match = self._selected_match()
        if match is None or not (match.error or "").strip():
            self.notify("No error text to explain yet.", severity="warning")
            return
        if match.id in self._explaining_ids:
            self.notify(f"Already asking the model about #{match.id}.")
            return
        self._explaining_ids.add(match.id)
        self.notify(f"Asking the model to explain #{match.id}... (can take a while for local models)")
        if self._selected_id == match.id:
            self.run_worker(self._show_selected(), exclusive=True, group="detail")
        # Grouped per error id (not one shared "explain" group): explaining a
        # different error must not cancel an unrelated one already in flight.
        self.run_worker(self._do_explain(match.id), exclusive=True, group=f"explain-{match.id}")

    async def _do_explain(self, error_id: int) -> None:
        from .explain import generate_explanation_verbose

        match = next((e for e in self._all_errors if e.id == error_id), None)
        if match is None:
            self._explaining_ids.discard(error_id)
            return
        try:
            explanation, detail = await run_in_daemon_thread(
                generate_explanation_verbose, match.failing_command, match.error, match.fix or None
            )
        except Exception as exc:  # noqa: BLE001 - run_in_daemon_thread itself never should, but stay defensive
            explanation, detail = None, f"{exc.__class__.__name__}: {exc}"
        finally:
            self._explaining_ids.discard(error_id)

        if explanation:
            self._store.set_ai_explanation(error_id, explanation)
            match.ai_explanation = explanation
            self.notify(f"Explanation ready for #{error_id}.")
        else:
            # Surface the *actual* failure (timeout, connection refused, bad
            # model name, ...) instead of a generic "didn't work" — this is
            # exactly the detail that was previously swallowed and impossible
            # to diagnose from the dashboard alone.
            self.notify(f"Explain failed for #{error_id}: {detail}", severity="error", timeout=10)
        if self._selected_id == error_id:
            await self._show_selected()


def main() -> None:
    fixerrDashboard().run()


if __name__ == "__main__":  # pragma: no cover
    main()
