"""Optional HTTP Basic auth + same-origin CSRF guard.

Disabled by default (auth_credentials="" → middleware passes everything
through) so first-boot UX is unchanged. Set BABEL_AUTH_CREDENTIALS to
"user:password" — or save the value via Settings — to require auth on
every endpoint except /health (which Docker healthchecks need).

CSRF: when auth is enabled, browsers attach the Basic credentials
automatically on cross-site form posts, so a malicious page on another
LAN host could trigger /api/process or /api/settings if it knew your
URL. We block that by checking on state-changing methods (POST/PATCH/
PUT/DELETE) that the request's Origin or Referer header matches the
configured Host header. Same-origin AJAX from the real UI passes; a
cross-site form post fails. /health is exempt because it has no
state-changing equivalent and may be called by orchestration tools that
don't set Origin.
"""
import base64
import secrets
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# Endpoints that bypass auth + CSRF entirely. /health is the Docker probe.
_AUTH_EXEMPT_PATHS: set[str] = {"/health"}

# State-changing methods that need the same-origin check when auth is on.
_STATE_CHANGING_METHODS: set[str] = {"POST", "PATCH", "PUT", "DELETE"}


def _unauthorized() -> Response:
    return Response(
        "Authentication required.",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Subtitle This"'},
    )


def _forbidden(reason: str) -> Response:
    return Response(reason, status_code=403)


class AuthAndCsrfMiddleware(BaseHTTPMiddleware):
    """Reads `auth_credentials` from settings on every request.

    Reading per-request lets users enable/disable auth via the Settings UI
    without restarting the container. The cost is negligible (one dict
    lookup) and the alternative — caching at startup — would surprise
    users when their Save click had no effect.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        # Lazy import: app.config imports pydantic which is heavier than
        # we want loaded for module-level middleware setup, and it lets
        # tests stub settings cleanly.
        from app.config import settings

        configured = (settings.auth_credentials or "").strip()
        if not configured:
            # Auth disabled — preserve the zero-config first-boot path.
            return await call_next(request)

        # ── HTTP Basic auth ──────────────────────────────────────────────
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("basic "):
            return _unauthorized()
        try:
            decoded = base64.b64decode(header[6:].strip(), validate=True).decode(
                "utf-8", errors="replace",
            )
        except (ValueError, UnicodeDecodeError):
            return _unauthorized()
        # secrets.compare_digest avoids a timing oracle on the password.
        if not secrets.compare_digest(decoded, configured):
            return _unauthorized()

        # ── CSRF: same-origin check on state-changing methods ────────────
        if request.method.upper() in _STATE_CHANGING_METHODS:
            if not _origin_matches_host(request):
                return _forbidden(
                    "Cross-origin request blocked. State-changing requests must "
                    "originate from the same host as this server."
                )

        return await call_next(request)


def _origin_matches_host(request: Request) -> bool:
    """Return True iff the request's Origin (or Referer fallback) header
    matches the Host header.

    Browsers always send Origin on POST from a cross-origin context (and
    on same-origin too in modern browsers). Some HTMX flows omit Origin
    on same-origin same-document POSTs; in that case we fall back to
    Referer. If neither is present, we accept the request — direct API
    clients (curl, custom scripts) authenticate via Basic and don't ride
    on a browser session, so the CSRF threat doesn't apply to them.
    """
    host = (request.headers.get("host") or "").strip().lower()
    if not host:
        # No Host header — be permissive (this is rare and not exploitable
        # without the attacker also having the credentials).
        return True

    origin = request.headers.get("origin")
    if origin:
        return _hostport_of(origin) == host

    referer = request.headers.get("referer")
    if referer:
        return _hostport_of(referer) == host

    # Browsers send at least one of these on cross-origin requests; their
    # absence indicates a non-browser client (curl, scripts) which we
    # accept on the strength of Basic auth alone.
    return True


def _hostport_of(url: str) -> str:
    """Extract `host:port` (lowercased, no scheme/path) from a URL.
    Returns "" if the URL is malformed."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    netloc = parts.netloc.lower()
    # Strip user-info if present (user@host) — only the host:port matters.
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]
    return netloc
