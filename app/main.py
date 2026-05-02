import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import jobs
from app.api.manage import router as manage_router
from app.api.settings_api import router as settings_router
from app.api.transcribe import router as transcribe_router
from app.api.webhook import router as webhook_router
from app.ui.routes import router as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Capture the main event loop so sync routes can schedule async jobs.
    jobs.set_main_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="Babel Tower", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(transcribe_router)
app.include_router(manage_router)
app.include_router(settings_router)
app.include_router(webhook_router)
app.include_router(ui_router)
