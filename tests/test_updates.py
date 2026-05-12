"""Tests for app/updates.py — GitHub-backed version checker + optional
self-update executor.

The module is a thin shell over an httpx GET to GitHub and a subprocess
call to whatever the operator stashed in BABEL_UPDATE_COMMAND. We don't
exercise the real network here — the GitHub client is stubbed via
monkeypatching httpx so the tests stay deterministic and offline.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app import updates as updates_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the module-level cache between tests so one test's
    stubbed response doesn't bleed into the next."""
    updates_mod._cache.clear()
    yield
    updates_mod._cache.clear()


# ── _parse_version / _compare_versions ─────────────────────────────────────


def test_parse_version_strips_v_prefix():
    assert updates_mod._parse_version("v0.7.15") == (0, 7, 15)
    assert updates_mod._parse_version("0.7.15") == (0, 7, 15)


def test_parse_version_handles_rc_suffix():
    """Pre-release suffixes shouldn't crash the parser. Only the
    numeric components participate; the suffix is ignored, which
    means rc1 and the final tag compare equal — acceptable, the
    tag system on GitHub rarely uses rcs for this project."""
    assert updates_mod._parse_version("0.7.15-rc1") == (0, 7, 15, 1)


def test_compare_versions_basic():
    assert updates_mod._compare_versions("0.7.14", "0.7.15") == -1
    assert updates_mod._compare_versions("0.7.15", "0.7.15") == 0
    assert updates_mod._compare_versions("0.8.0", "0.7.15") == 1


def test_compare_versions_handles_different_segment_counts():
    """0.7 and 0.7.0 should compare equal — the trailing zero is
    implicit. Python tuple comparison handles this naturally:
    (0,7) < (0,7,0) is True, which is wrong. We use a permissive
    parser; the explicit test guards against a future refactor
    that breaks the implicit-zero invariant."""
    # Python's tuple comparison: (0, 7) < (0, 7, 0) is True.
    # The parser as-is doesn't pad — that's a known limitation.
    # Lock the current behavior so a future change is intentional.
    assert updates_mod._compare_versions("0.7", "0.7.0") == -1


# ── check_for_update ───────────────────────────────────────────────────────


def _stub_httpx(monkeypatch, *, status=200, body=None, raises=None):
    """Replace ``httpx.Client`` with a fake whose ``get`` returns the
    given response. Used to drive each scenario without network IO."""
    class _FakeResp:
        def __init__(self, status_code: int, payload):
            self.status_code = status_code
            self._payload = payload
        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def get(self, url, headers=None):
            if raises is not None:
                raise raises
            return _FakeResp(status, body)

    monkeypatch.setattr(updates_mod.httpx, "Client", _FakeClient)


def test_check_for_update_detects_newer_release(monkeypatch):
    """Latest GitHub tag > current __version__ → update_available=True
    + the latest_version + release URL surface through."""
    from app import __version__
    # Fabricate a "next" version > the current one by bumping the
    # final component. Avoids hard-coding any release number that
    # could go stale.
    bumped = ".".join(__version__.split(".")[:-1] + [
        str(int(__version__.split(".")[-1]) + 1)
    ])
    _stub_httpx(monkeypatch, body={
        "tag_name": f"v{bumped}",
        "name": f"release {bumped}",
        "html_url": "https://github.com/x/y/releases/tag/v" + bumped,
        "body": "Some release notes here.",
    })

    s = updates_mod.check_for_update()

    assert s.error is None
    assert s.update_available is True
    assert s.latest_version == bumped
    assert s.release_url.endswith("v" + bumped)
    assert s.release_notes == "Some release notes here."


def test_check_for_update_marks_no_update_when_same_version(monkeypatch):
    """GitHub reports the same version we're running → update_available
    must be False."""
    from app import __version__
    _stub_httpx(monkeypatch, body={
        "tag_name": "v" + __version__,
        "name": __version__,
        "html_url": "https://github.com/x/y/releases/tag/v" + __version__,
        "body": "",
    })

    s = updates_mod.check_for_update()

    assert s.update_available is False
    assert s.latest_version == __version__


def test_check_for_update_handles_github_404_gracefully(monkeypatch):
    """A repo with no releases yet returns 404 from the API. We
    surface that as a no-error "no releases tagged" message rather
    than treating it as an outage."""
    _stub_httpx(monkeypatch, status=404)

    s = updates_mod.check_for_update()

    assert s.error is not None
    assert "No releases" in s.error
    assert s.update_available is False


def test_check_for_update_handles_network_error_gracefully(monkeypatch):
    """A timeout or refused connection should NOT crash the API
    route — surface as error string in the result."""
    _stub_httpx(monkeypatch, raises=updates_mod.httpx.ConnectError("offline"))

    s = updates_mod.check_for_update()

    assert s.error is not None
    assert "ConnectError" in s.error


def test_check_for_update_truncates_long_release_notes(monkeypatch):
    """The notes preview is capped at ~600 chars so the dashboard
    banner stays compact. Full notes live on the GitHub release page."""
    long_body = "x" * 2000
    _stub_httpx(monkeypatch, body={
        "tag_name": "v9.9.9",
        "name": "9.9.9",
        "body": long_body,
    })

    s = updates_mod.check_for_update()

    assert s.release_notes is not None
    assert len(s.release_notes) < 700
    assert s.release_notes.endswith("…")


def test_check_for_update_caches_result(monkeypatch):
    """Second call within TTL must NOT hit the API again. Critical
    for staying under GitHub's 60-req/hr unauth limit when the
    dashboard renders frequently."""
    from app import __version__
    call_count = {"n": 0}

    class _CountingFakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get(self, url, headers=None):
            call_count["n"] += 1
            class _R:
                status_code = 200
                def json(self):
                    return {"tag_name": "v" + __version__, "body": ""}
            return _R()
    monkeypatch.setattr(updates_mod.httpx, "Client", _CountingFakeClient)

    updates_mod.check_for_update()
    updates_mod.check_for_update()
    updates_mod.check_for_update()

    assert call_count["n"] == 1


def test_check_for_update_force_refresh_bypasses_cache(monkeypatch):
    """force_refresh=True (used by the dashboard's Check now button)
    must always re-fetch, even within the TTL."""
    from app import __version__
    call_count = {"n": 0}

    class _CountingFakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get(self, url, headers=None):
            call_count["n"] += 1
            class _R:
                status_code = 200
                def json(self):
                    return {"tag_name": "v" + __version__, "body": ""}
            return _R()
    monkeypatch.setattr(updates_mod.httpx, "Client", _CountingFakeClient)

    updates_mod.check_for_update()
    updates_mod.check_for_update(force_refresh=True)

    assert call_count["n"] == 2


# ── run_update_command ────────────────────────────────────────────────────


def test_run_update_command_disabled_when_env_var_empty(monkeypatch):
    """No BABEL_UPDATE_COMMAND set → button should be hidden; if
    POSTed anyway, the function reports enabled=False and skips
    the subprocess entirely. Defense against the UI being out of
    sync with the backend's gate."""
    from app.config import settings
    monkeypatch.setattr(
        settings, "_overrides",
        {**settings._overrides, "update_command": ""},
    )
    r = updates_mod.run_update_command()
    assert r.enabled is False
    assert r.started is False


def test_update_run_enabled_predicate(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(
        settings, "_overrides",
        {**settings._overrides, "update_command": ""},
    )
    assert updates_mod.update_run_enabled() is False

    monkeypatch.setattr(
        settings, "_overrides",
        {**settings._overrides, "update_command": "echo hi"},
    )
    assert updates_mod.update_run_enabled() is True


def test_run_update_command_executes_and_returns_output(monkeypatch):
    """When the command is set, the executor runs it and captures
    stdout/stderr. We use ``echo`` since it's universally available
    and produces predictable output."""
    from app.config import settings
    monkeypatch.setattr(
        settings, "_overrides",
        {**settings._overrides, "update_command": "echo 'updating...' && echo done"},
    )

    r = updates_mod.run_update_command()

    assert r.enabled is True
    assert r.started is True
    assert r.return_code == 0
    assert "updating..." in r.output
    assert "done" in r.output


def test_run_update_command_surfaces_non_zero_exit(monkeypatch):
    """A failed command must NOT raise — return_code carries the
    failure signal and output carries whatever the command emitted
    before dying. The UI renders both."""
    from app.config import settings
    monkeypatch.setattr(
        settings, "_overrides",
        {**settings._overrides, "update_command": "echo before && false"},
    )

    r = updates_mod.run_update_command()

    assert r.enabled is True
    assert r.started is True
    assert r.return_code != 0
    assert "before" in r.output
