"""GitHub-backed update checker + optional self-update executor.

Two surfaces:

- ``check_for_update()`` queries the GitHub Releases API for the
  repo's latest tagged release, compares it to the running app's
  ``__version__``, and returns a structured result the UI renders
  as either a "you're up to date" indicator or an actionable
  "update available" banner. Read-only, no credentials required,
  results cached for 1 hour to stay well under GitHub's
  unauthenticated rate limit (60 req/hr/IP).

- ``run_update_command()`` invokes whatever shell command the
  operator stashed in the ``BABEL_UPDATE_COMMAND`` env var (mapped
  to ``settings.update_command``). When empty (default) the
  executor is disabled — the UI hides the "Update now" button and
  the user copy-pastes the suggested commands by hand. When set,
  the button is exposed and clicking it streams the command's
  output back to the page. The env-var-only requirement is the
  security wall: there's no user-supplied path that could turn
  this into arbitrary command injection.

Self-update from inside a Docker container has fundamental limits.
This module deliberately does NOT try to:

- Detect whether the container is using ``build:`` vs ``image:`` —
  the operator's command knows which flow applies.
- Mount the docker socket — that's a container-level choice we
  can't make for the user.
- Inject a watchtower-style supervisor — that's the right tool
  but an external setup, not something an app can install for itself.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx


_log = logging.getLogger("subtitle_this")
_GITHUB_REPO = "damienigg/subtitle-this"
_GITHUB_RELEASES_URL = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"
_RELEASE_PAGE_URL = f"https://github.com/{_GITHUB_REPO}/releases/latest"

# Cache the GitHub result so the dashboard's status pill doesn't hit
# the API on every page load. 1 hour matches a typical "I'll check at
# most once an hour" cadence — GitHub's unauthenticated limit is 60/hr
# so even a polling UI stays well below.
_CACHE_TTL_SECONDS = 3600
_cache_lock = threading.Lock()
_cache: dict[str, Any] = {}


@dataclass
class UpdateStatus:
    """Surface returned to the API + template. Each field is JSON-safe
    so the same struct doubles as the API response payload."""
    current_version: str
    latest_version: str | None = None
    update_available: bool = False
    release_url: str = _RELEASE_PAGE_URL
    release_name: str | None = None
    # Truncated release notes (first ~600 chars of the body). Full
    # notes live on GitHub at release_url.
    release_notes: str | None = None
    # When the check completed (epoch seconds). The UI surfaces it as
    # "Checked X minutes ago" so the user can tell freshness apart
    # from up-to-date-ness.
    checked_at_epoch: float | None = None
    # When the check FAILED (network down, GitHub 5xx, rate-limited),
    # error carries the reason and the other fields hold stale-or-
    # empty values. Keeps the UI from rendering "you're up to date"
    # on a check that simply couldn't reach GitHub.
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_version(s: str) -> tuple[int, ...]:
    """Permissive semver-ish parser. Accepts ``v0.7.15`` and
    ``0.7.15-rc1`` alike; only the numeric components participate in
    the comparison. Anything unrecognized maps to ``(0,)`` so the
    comparison falls through to "older" rather than crashing."""
    if not s:
        return (0,)
    s = s.lstrip("v")
    parts = re.split(r"[^\d]+", s)
    return tuple(int(p) for p in parts if p.isdigit()) or (0,)


def _compare_versions(a: str, b: str) -> int:
    """Return -1 if a < b, 0 if equal, +1 if a > b."""
    pa, pb = _parse_version(a), _parse_version(b)
    if pa < pb:
        return -1
    if pa > pb:
        return 1
    return 0


def check_for_update(force_refresh: bool = False) -> UpdateStatus:
    """Hit the GitHub Releases API (cached for 1 h) and return the
    comparison. Never raises — network / API errors surface as
    ``error`` on the returned UpdateStatus so the UI can render a
    "couldn't check" message rather than a 500."""
    from app import __version__
    now = time.time()
    with _cache_lock:
        cached = _cache.get("result")
        cached_at = _cache.get("checked_at", 0)
    if not force_refresh and cached and (now - cached_at) < _CACHE_TTL_SECONDS:
        # Return the cached copy. The age is reflected on the UI via
        # the checked_at_epoch field so the operator can tell stale
        # answers apart from fresh ones.
        return UpdateStatus(**cached)

    status = UpdateStatus(current_version=__version__, checked_at_epoch=now)
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            r = client.get(
                _GITHUB_RELEASES_URL,
                headers={"Accept": "application/vnd.github+json"},
            )
        if r.status_code == 404:
            # The repo has never tagged a release. Not an error per
            # se — just report no latest version.
            status.error = "No releases tagged on GitHub yet."
        elif r.status_code != 200:
            status.error = f"GitHub returned HTTP {r.status_code}"
        else:
            data = r.json()
            tag = data.get("tag_name") or data.get("name") or ""
            status.latest_version = tag.lstrip("v") or None
            status.release_name = data.get("name") or status.latest_version
            status.release_url = data.get("html_url") or _RELEASE_PAGE_URL
            body = (data.get("body") or "").strip()
            if body:
                # Truncate to a readable preview. The "..." marker
                # tells the user there's more on the release page.
                status.release_notes = body[:600] + ("…" if len(body) > 600 else "")
            if status.latest_version:
                status.update_available = (
                    _compare_versions(__version__, status.latest_version) < 0
                )
    except (httpx.HTTPError, ValueError) as e:
        status.error = f"{type(e).__name__}: {e}"

    with _cache_lock:
        _cache["result"] = status.to_dict()
        _cache["checked_at"] = now
    return status


@dataclass
class UpdateRunResult:
    """Output of ``run_update_command``. ``output`` is the combined
    stdout+stderr; the UI renders it in a monospace block so the
    operator can see exactly what the update did."""
    enabled: bool
    started: bool
    return_code: int | None = None
    output: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_update_command() -> UpdateRunResult:
    """Execute ``settings.update_command`` via the shell and capture
    the output. Disabled (returns ``enabled=False``) when the env var
    isn't set — the UI shouldn't have offered the button in that case
    but we re-check defensively.

    The command runs with the container's privileges. If you want it
    to invoke ``docker compose``, mount the docker socket AND make
    sure the container's image has the ``docker`` CLI installed (the
    openvino base does not by default — set ``BABEL_UPDATE_COMMAND``
    to a ``docker pull`` + Python self-exec sequence, or use an
    external orchestrator like watchtower instead).

    No streaming yet — output is buffered to the end of the run, then
    returned all at once. A long update (several minutes for an image
    pull) keeps the HTTP request open; the UI shows a "Updating…"
    spinner over that time. Acceptable for v1; if it becomes a UX
    issue, the upgrade path is server-sent events / polling.
    """
    from app.config import settings
    cmd = (settings.update_command or "").strip()
    if not cmd:
        return UpdateRunResult(enabled=False, started=False,
                               error="BABEL_UPDATE_COMMAND is not set.")
    import subprocess
    try:
        # shell=True is intentional — the operator's command is a
        # composite (cd … && git pull && docker compose up). We're
        # NOT taking user-supplied input here; the command came from
        # the env var which is operator-controlled by definition.
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=15 * 60,   # 15-minute ceiling; pulls + builds fit
        )
        output = (proc.stdout or "") + (
            ("\n--- stderr ---\n" + proc.stderr) if proc.stderr else ""
        )
        return UpdateRunResult(
            enabled=True, started=True,
            return_code=proc.returncode,
            output=output,
        )
    except subprocess.TimeoutExpired as e:
        return UpdateRunResult(
            enabled=True, started=True, return_code=None,
            output=(e.stdout or "") + (e.stderr or ""),
            error=f"Timed out after {e.timeout:.0f}s",
        )
    except OSError as e:
        return UpdateRunResult(
            enabled=True, started=False,
            error=f"Failed to launch update command: {e}",
        )


def update_run_enabled() -> bool:
    """Convenience predicate the template uses to decide whether to
    show the "Update now" button. False when ``update_command`` is
    empty or whitespace-only."""
    from app.config import settings
    return bool((settings.update_command or "").strip())
