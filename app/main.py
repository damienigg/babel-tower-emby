import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__, jobs
from app.api.manage import router as manage_router
from app.api.settings_api import router as settings_router
from app.auth import AuthAndCsrfMiddleware
from app.ui.routes import router as ui_router


# Cap the event loop's default ThreadPoolExecutor at 4 workers (matching
# `cpus: "4.0"` in docker-compose.yml). Python's default sizes the pool
# at min(32, os.cpu_count()+4), which on a 16-core TrueNAS host is 20 —
# under HTMX poll bursts and concurrent /api/server/items lookups, the
# default executor could spawn ~20 threads each running OMP-parallel
# torch/numpy code, blowing past the cgroup CPU cap. The job lock
# already serializes the heavy pipeline; this cap protects the sync
# FastAPI handlers that aren't gated by it (settings PATCH, server
# probes, library listing).
_EXECUTOR_MAX_WORKERS = 4


# Surface our INFO-level logs (e.g. the [openvino] device-selection line) in
# `docker logs`. Uvicorn doesn't propagate non-uvicorn loggers by default at
# INFO, so we wire up a basic stderr handler if nothing else has.
_pkg_logger = logging.getLogger("subtitle_this")
if not _pkg_logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _pkg_logger.addHandler(h)
_pkg_logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Capture the main event loop so sync routes can schedule async jobs.
    loop = asyncio.get_running_loop()
    jobs.set_main_loop(loop)
    # Replace the default ThreadPoolExecutor with a bounded one so
    # asyncio.to_thread + sync route handlers can't oversubscribe the
    # CPU under burst load. The executor is what receives every
    # `asyncio.to_thread` call AND every sync FastAPI route's threadpool
    # dispatch — capping it puts an upper bound on concurrent OMP-parallel
    # torch/numpy work inside the container.
    loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=_EXECUTOR_MAX_WORKERS,
            thread_name_prefix="subtitle-this-worker",
        )
    )
    yield


# Subtitle creation is exclusively a manual user action through the web UI,
# and only ever per-item or per-batch — never library-wide. We deliberately
# do NOT expose:
# - a webhook receiver (no auto-triggering on item-added events)
# - a path-based /transcribe-translate endpoint (no curl-driven workflow)
# - a sweep-everything endpoint (no whole-library subtitling — too much
#   spend potential, and there's no real use case that "subtitle every
#   item in my 5000-film library" addresses better than a deliberate
#   batch selection)
# The endpoints registered below back the per-item "Subtitle this" button
# and the multi-select batch flow on the Library page, plus the auto-
# refreshing jobs list — they're not meant as a public CLI surface.
app = FastAPI(title="Subtitle This", version=__version__, lifespan=lifespan)

# Auth + CSRF guard. No-op when settings.auth_credentials is unset — that's
# the default, so the existing zero-config first-boot experience is
# preserved. Setting BABEL_AUTH_CREDENTIALS="user:pass" (or saving the
# value via the Settings UI) immediately gates every endpoint except
# /health, and adds a same-origin check on POST/PATCH/PUT/DELETE so a
# cross-site CSRF page can't ride your browser's stored Basic creds.
app.add_middleware(AuthAndCsrfMiddleware)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/version")
def api_version() -> dict:
    """Cheap programmatic version probe. Same string the footer renders,
    same one FastAPI's OpenAPI doc reports — single source of truth in
    app/__init__.py:__version__."""
    return {"version": __version__}


app.include_router(manage_router)
app.include_router(settings_router)
app.include_router(ui_router)
