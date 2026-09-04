"""Tests for the `fixerr dashboard` Textual TUI.

Embeddings are mocked (deterministic fixed vectors) so these never need a
running Ollama — same convention as test_ai_backends.py. Each test wraps its
async body in ``asyncio.run`` rather than pulling in pytest-asyncio, since
that's the only dependency it would add.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from fixerr import dashboard as dashboard_module
from fixerr.dashboard import compute_clusters, fixerrDashboard, kmeans
from fixerr.store import STATUS_OPEN, STATUS_RESOLVED, STATUS_WONT_FIX, Store


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("fixerr_DB", str(tmp_path / "errors.db"))
    monkeypatch.setenv("fixerr_CONFIG", str(tmp_path / "config.toml"))
    # Deterministic, no-network embeddings: hash the text into a small vector.
    def fake_embed(text: str):
        h = sum(text.encode())
        return [(h % (i + 7)) / 10.0 for i in range(8)]

    fake_backend = mock.Mock()
    fake_backend.embed.side_effect = fake_embed
    fake_backend.generate.return_value = "mock response"
    monkeypatch.setattr("fixerr.store.get_embed_backend", lambda: fake_backend)
    # The dashboard's provider-status probes (_provider_status /
    # _probe_embed_offline) hold their own references to get_backend /
    # get_embed_backend — mock those too, or every test silently makes a
    # real (possibly slow, model-dependent) network call on mount.
    monkeypatch.setattr("fixerr.dashboard.get_backend", lambda: fake_backend)
    monkeypatch.setattr("fixerr.ai.get_embed_backend", lambda: fake_backend)
    return Store()


def seed(store, n=4):
    ids = []
    samples = [
        ("docker compose up", "bind for 0.0.0.0:5432 failed: port already in use", 1),
        ("pytest tests/", "ModuleNotFoundError: No module named 'httpx'", 1),
        ("npm run build", "network error while fetching docker image", 1),
        ("./deploy.sh", "permission denied: /var/run/docker.sock", 126),
    ]
    for i in range(n):
        cmd, err, code = samples[i % len(samples)]
        ids.append(store.add_error(f"{cmd} #{i}", f"{err} on service {i}", exit_code=code))
    return ids


# --------------------------------------------------------------- data load ---

def test_loads_real_data_and_focuses_list(store):
    seed(store, 4)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            assert app.focused is not None and app.focused.id == "error-list"
            assert len(app._all_errors) == 4
            assert app._selected_id == app._all_errors[0].id

    run(body())


def test_empty_db_shows_onboarding(store):
    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            assert len(app.query(".onboarding")) == 1

    run(body())


# ------------------------------------------------------------------ filters ---

def test_tab_filtering_by_status(store):
    ids = seed(store, 4)
    store.set_fix(ids[0], "fixed it")
    store.dismiss(ids[1])

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app.action_tab_unresolved()
            await pilot.pause(0.1)
            assert all(e.status == STATUS_OPEN for e in app._visible)
            app.action_tab_resolved()
            await pilot.pause(0.1)
            assert all(e.status == STATUS_RESOLVED for e in app._visible)
            app.action_tab_all()
            await pilot.pause(0.1)
            assert len(app._visible) == 4

    run(body())


def test_tag_chip_click_filters(store):
    seed(store, 4)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            chips = app.query_one("FilterChips")
            docker_chip = next(c for c in chips.query("_Chip") if c.tag_name == "docker")
            await pilot.click(docker_chip)
            await pilot.pause(0.1)
            assert app._visible and all("docker" in dashboard_module.derive_tags(e) for e in app._visible)
            all_chip = next(c for c in chips.query("_Chip") if c.tag_name == "all")
            await pilot.click(all_chip)
            await pilot.pause(0.1)
            assert len(app._visible) == 4

    run(body())


# ---------------------------------------------------------------- mutations ---

def test_resolve_dismiss_copy_actions(store):
    ids = seed(store, 2)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app._selected_id = ids[0]

            app.action_mark_resolved()
            await pilot.pause(0.1)
            assert store.get(ids[0])["status"] == STATUS_RESOLVED

            # Resolving again is a no-op (guarded), doesn't clear resolved_at.
            resolved_at = store.get(ids[0])["resolved_at"]
            app.action_mark_resolved()
            await pilot.pause(0.1)
            assert store.get(ids[0])["resolved_at"] == resolved_at

            app._selected_id = ids[1]
            app.action_dismiss()
            await pilot.pause(0.1)
            assert store.get(ids[1])["status"] == STATUS_WONT_FIX

    run(body())


def test_copy_to_system_clipboard_uses_pbcopy_on_macos(monkeypatch):
    from unittest import mock

    import fixerr.dashboard as dashboard_module

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    run_mock = mock.Mock()
    monkeypatch.setattr("subprocess.run", run_mock)

    ok = dashboard_module.copy_to_system_clipboard("some fix text")
    assert ok is True
    args, kwargs = run_mock.call_args
    assert args[0] == ["pbcopy"]
    assert kwargs["input"] == b"some fix text"


def test_copy_to_system_clipboard_returns_false_when_no_tool_available(monkeypatch):
    import fixerr.dashboard as dashboard_module

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("subprocess.run", mock.Mock(side_effect=FileNotFoundError))

    assert dashboard_module.copy_to_system_clipboard("text") is False


def test_action_copy_fix_uses_native_clipboard_when_available(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "the real fix text")

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app._selected_id = eid

            with mock.patch("fixerr.dashboard.copy_to_system_clipboard", return_value=True) as native, \
                 mock.patch.object(app, "copy_to_clipboard") as osc52:
                app.action_copy_fix()
                native.assert_called_once_with("the real fix text")
                osc52.assert_not_called()  # native succeeded, no need for the fallback

    run(body())


def test_action_copy_fix_falls_back_to_osc52_when_no_native_tool(store):
    eid = store.add_error("docker compose up", "port already in use", exit_code=1)
    store.set_fix(eid, "the real fix text")

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app._selected_id = eid

            with mock.patch("fixerr.dashboard.copy_to_system_clipboard", return_value=False), \
                 mock.patch.object(app, "copy_to_clipboard") as osc52:
                app.action_copy_fix()
                osc52.assert_called_once_with("the real fix text")

    run(body())


def test_action_reopen_reverts_resolved_and_wont_fix_to_open(store):
    ids = seed(store, 2)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)

            app._selected_id = ids[0]
            app.action_mark_resolved()
            await pilot.pause(0.1)
            assert store.get(ids[0])["status"] == STATUS_RESOLVED

            app._selected_id = ids[0]
            app.action_reopen()
            await pilot.pause(0.1)
            row = store.get(ids[0])
            assert row["status"] == STATUS_OPEN
            assert row["resolved_at"] is None

            # Reopening an already-open error is a no-op (guarded).
            app.action_reopen()
            await pilot.pause(0.1)
            assert store.get(ids[0])["status"] == STATUS_OPEN

            app._selected_id = ids[1]
            app.action_dismiss()
            await pilot.pause(0.1)
            app.action_reopen()
            await pilot.pause(0.1)
            assert store.get(ids[1])["status"] == STATUS_OPEN

    run(body())


def test_action_explain_generates_and_refreshes_detail(store, monkeypatch):
    from unittest import mock

    ids = seed(store, 1)
    fake_backend = mock.Mock(generate=mock.Mock(return_value="Because of a port conflict."))
    monkeypatch.setattr("fixerr.explain.get_backend", lambda: fake_backend)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app._selected_id = ids[0]
            app.action_explain()
            await pilot.pause(0.3)  # worker runs generate_explanation in a thread
            assert store.get(ids[0])["ai_explanation"] == "Because of a port conflict."

    run(body())


def test_action_explain_shows_persistent_in_progress_state(store, monkeypatch):
    """The toast notification disappears in a few seconds, but local models
    can take much longer — the detail panel itself must show an ongoing
    "asking the model" state for as long as generation is actually running,
    not just a transient toast that vanishes long before it's done."""
    import time
    from unittest import mock

    ids = seed(store, 1)

    def slow_generate(*a, **kw):
        time.sleep(0.4)
        return "slow explanation"

    fake_backend = mock.Mock(generate=mock.Mock(side_effect=slow_generate))
    monkeypatch.setattr("fixerr.explain.get_backend", lambda: fake_backend)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app._selected_id = ids[0]
            await app._show_selected()

            app.action_explain()
            await pilot.pause(0.1)  # generation still running (sleeps 0.4s)
            assert ids[0] in app._explaining_ids
            ai_widget = app.query_one("#detail-ai")
            assert "asking the model" in str(ai_widget.render()).lower()

            await pilot.pause(0.5)  # let it finish
            assert ids[0] not in app._explaining_ids
            ai_widget = app.query_one("#detail-ai")
            assert "slow explanation" in str(ai_widget.render())

    run(body())


def test_action_explain_second_error_does_not_cancel_first(store, monkeypatch):
    """Explaining a different error while one is already in flight must not
    cancel the first (a shared worker "group" would do exactly that)."""
    import time
    from unittest import mock

    ids = seed(store, 2)
    calls = []

    def slow_generate(prompt):
        calls.append(prompt)
        time.sleep(0.3)
        return f"explained ({len(calls)})"

    fake_backend = mock.Mock(generate=mock.Mock(side_effect=slow_generate))
    monkeypatch.setattr("fixerr.explain.get_backend", lambda: fake_backend)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)

            app._selected_id = ids[0]
            app.action_explain()
            await pilot.pause(0.05)
            app._selected_id = ids[1]
            app.action_explain()
            await pilot.pause(0.5)  # both should finish

            assert store.get(ids[0])["ai_explanation"] is not None
            assert store.get(ids[1])["ai_explanation"] is not None

    run(body())


def test_exit_does_not_block_on_a_pending_explain_worker(store, monkeypatch):
    """Regression test: quitting used to hang if a slow generate() call was
    still in flight, because asyncio.to_thread's default executor threads are
    non-daemon and concurrent.futures joins them at interpreter exit. See
    dashboard.run_in_daemon_thread."""
    import time

    ids = seed(store, 1)

    def slow_generate(*a, **kw):
        time.sleep(2)  # long enough that a real hang would fail this test
        return "explained"

    fake_backend = mock.Mock(generate=mock.Mock(side_effect=slow_generate))
    monkeypatch.setattr("fixerr.explain.get_backend", lambda: fake_backend)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app._selected_id = ids[0]
            app.action_explain()
            await pilot.pause(0.3)  # worker has started, is now sleeping for 2s

            t0 = time.time()
            app.exit()
            await pilot.pause(0.05)
            elapsed = time.time() - t0
            assert elapsed < 1.0, f"exit() took {elapsed:.2f}s — blocked on the pending worker"

    run(body())


# ------------------------------------------------------------------- search ---

def test_search_filters_when_embeddings_offline(store, monkeypatch):
    seed(store, 4)

    async def body():
        app = fixerrDashboard(store=store)
        monkeypatch.setattr(app, "_embed_offline", True)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app._embed_offline = True  # reload_data's probe would flip this back; pin it
            app._search_query = "docker"
            await app._apply_filters()
            assert app._visible
            assert all(
                "docker" in (e.error or "").lower() or "docker" in (e.failing_command or "").lower()
                for e in app._visible
            )

    run(body())


# ------------------------------------------------------------ similar errors ---

def test_similar_errors_excludes_self(store):
    ids = seed(store, 4)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.2)
            app._selected_id = ids[0]
            await app._show_selected()
            await pilot.pause(0.2)
            rows = list(app.query_one("#detail-similar").query("_SimilarRow"))
            matched_ids = [r.match_id for r in rows]
            assert ids[0] not in matched_ids, f"ids[0]={ids[0]}, matched_ids={matched_ids}"

    run(body())


def test_similar_errors_hidden_when_offline(store):
    ids = seed(store, 4)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app._embed_offline = True
            app._selected_id = ids[0]
            await app._show_selected()
            await pilot.pause(0.1)
            assert app.query_one("#similar-label").display is False
            assert app.query_one("#detail-similar").display is False

    run(body())


# --------------------------------------------------------------- patterns ---

def test_patterns_gate_below_20(store):
    seed(store, 5)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            app.action_tab_patterns()
            await pilot.pause(0.1)
            assert app._patterns_cache is None

    run(body())


def test_patterns_computed_once_and_cached(store):
    seed(store, 24)

    async def body():
        app = fixerrDashboard(store=store)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(0.1)
            with mock.patch(
                "fixerr.dashboard.compute_clusters", side_effect=compute_clusters
            ) as spy:
                app.action_tab_patterns()
                await pilot.pause(0.1)
                app.action_tab_all()
                await pilot.pause(0.1)
                app.action_tab_patterns()
                await pilot.pause(0.1)
                assert spy.call_count == 1
            assert app._patterns_cache is not None
            assert sum(c.count for c in app._patterns_cache) == 24

    run(body())


def test_kmeans_separates_distinct_clusters():
    vectors = (
        [[1.0, 0.0, 0.0]] * 3
        + [[0.0, 1.0, 0.0]] * 3
        + [[0.0, 0.0, 1.0]] * 3
    )
    assignments = kmeans(vectors, k=3)
    assert assignments[0:3] == [assignments[0]] * 3
    assert assignments[3:6] == [assignments[3]] * 3
    assert assignments[6:9] == [assignments[6]] * 3
    assert len(set(assignments)) == 3
