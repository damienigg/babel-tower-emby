"""Settings GET/PATCH/DELETE. Sensitive fields are masked on read; on write
they're persisted to /cache/settings.json alongside everything else."""
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import READ_ONLY_FIELDS, SENSITIVE_FIELDS, settings


router = APIRouter(prefix="/api/settings")


@router.get("")
def get_settings() -> dict[str, Any]:
    """Return effective settings (env + overrides), with sensitive fields masked."""
    return {
        "values": settings.all_values(mask_sensitive=True),
        "sensitive": sorted(SENSITIVE_FIELDS),
        "read_only": sorted(READ_ONLY_FIELDS),
    }


@router.patch("")
def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected a JSON object of {field: value} pairs")
    # Strip sensitive empty strings — don't blank out an already-set key by accident
    cleaned = {k: v for k, v in payload.items() if not (k in SENSITIVE_FIELDS and v in ("", None))}
    try:
        settings.update(cleaned)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"values": settings.all_values(mask_sensitive=True)}


@router.delete("/{key}")
def reset_setting(key: str) -> dict[str, Any]:
    """Drop the user override for one field, reverting to the env-bound default."""
    settings.reset(key)
    return {"values": settings.all_values(mask_sensitive=True)}


@router.delete("")
def reset_all() -> dict[str, Any]:
    settings.reset_all()
    return {"values": settings.all_values(mask_sensitive=True)}
