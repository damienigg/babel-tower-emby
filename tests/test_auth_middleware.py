"""Tests for the optional HTTP Basic auth + same-origin CSRF middleware.

Auth is OFF by default — these tests confirm both ends of that switch
behave correctly:
- with `auth_credentials` empty: every endpoint passes through unchanged,
  preserving the zero-config first-boot experience the README promises.
- with `auth_credentials` set: protected endpoints require Basic, and
  POST/PATCH/DELETE/PUT additionally require same-origin (Origin or
  Referer must match Host).
- /health is always exempt so Docker healthchecks work without
  credentials.
"""
import base64

import pytest
from fastapi.testclient import TestClient

from app.config import settings as runtime_settings
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_auth(monkeypatch):
    """Each test starts with auth disabled. Tests that turn it on patch
    _overrides locally; this fixture guarantees no cross-test bleed."""
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": None},
    )


def _basic(user_pass: str) -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(user_pass.encode()).decode()}


# ── Auth OFF (default) ──────────────────────────────────────────────────────


def test_auth_off_lets_everything_through(client):
    r = client.get("/")
    assert r.status_code == 200, "dashboard should render with auth disabled"
    r = client.get("/api/settings")
    assert r.status_code == 200


def test_health_always_passes_without_auth(client, monkeypatch):
    """Docker / TrueNAS healthchecks hit /health without credentials —
    even when auth is enabled, /health MUST stay open."""
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": "user:pass"},
    )
    r = client.get("/health")
    assert r.status_code == 200


# ── Auth ON ─────────────────────────────────────────────────────────────────


def test_protected_endpoint_401s_without_credentials(client, monkeypatch):
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": "alice:secret"},
    )
    r = client.get("/api/settings")
    assert r.status_code == 401
    assert "Basic" in r.headers.get("www-authenticate", "")


def test_protected_endpoint_401s_with_wrong_password(client, monkeypatch):
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": "alice:secret"},
    )
    r = client.get("/api/settings", headers=_basic("alice:wrong"))
    assert r.status_code == 401


def test_protected_endpoint_200s_with_correct_credentials(client, monkeypatch):
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": "alice:secret"},
    )
    r = client.get("/api/settings", headers=_basic("alice:secret"))
    assert r.status_code == 200


def test_post_blocked_when_origin_mismatches_host(client, monkeypatch):
    """A cross-origin form post (CSRF attempt) with a stolen browser session
    must be rejected even with valid Basic credentials."""
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": "alice:secret"},
    )
    headers = _basic("alice:secret")
    headers["origin"] = "http://attacker.example.com"
    headers["host"] = "subtitle-this.lan"
    r = client.patch(
        "/api/settings",
        headers=headers,
        json={"max_line_chars": 50},
    )
    assert r.status_code == 403


def test_post_passes_when_origin_matches_host(client, monkeypatch):
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": "alice:secret"},
    )
    headers = _basic("alice:secret")
    # Origin == Host scheme stripped to netloc form.
    headers["origin"] = "http://subtitle-this.lan"
    headers["host"] = "subtitle-this.lan"
    r = client.patch(
        "/api/settings",
        headers=headers,
        json={"max_line_chars": 50},
    )
    assert r.status_code == 200


def test_get_with_referer_mismatch_still_passes(client, monkeypatch):
    """CSRF check applies only to state-changing methods (POST/PATCH/PUT/
    DELETE). A cross-origin GET is harmless on its own — browsers won't
    leak the response cross-origin."""
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": "alice:secret"},
    )
    headers = _basic("alice:secret")
    headers["referer"] = "http://attacker.example.com/some-page"
    r = client.get("/api/settings", headers=headers)
    assert r.status_code == 200


def test_post_without_origin_or_referer_passes(client, monkeypatch):
    """Direct API clients (curl, scripts) authenticate via Basic and don't
    set Origin/Referer. The CSRF threat doesn't apply to them — block only
    requests that demonstrably came from a browser on the wrong origin."""
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "auth_credentials": "alice:secret"},
    )
    r = client.patch(
        "/api/settings",
        headers=_basic("alice:secret"),
        json={"max_line_chars": 50},
    )
    assert r.status_code == 200
