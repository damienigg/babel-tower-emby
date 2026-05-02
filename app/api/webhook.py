"""Emby webhook receiver. Configure in Emby admin → Server Settings → Notifications
→ Add → Webhook with URL http://<host>:8765/webhook/emby. Set the
`webhook_secret` setting (and pass it as an `X-Babel-Token` header in the Emby
config) if you want shared-secret protection.

Emby's webhook payload varies by template; we accept arbitrary JSON and look for
the item ID in the most common locations: Item.Id, ItemId, id (lower/upper).
"""
from fastapi import APIRouter, HTTPException, Request

from app.api.manage import emby_client, submit_item_job
from app.config import settings
from app.emby.client import EmbyError


router = APIRouter()


_ADD_EVENT_KEYWORDS = ("itemadded", "library.new", "library.added")
_VIDEO_TYPES = {"movie", "episode", "video"}


def _is_added_event(event: str) -> bool:
    e = event.lower()
    return any(k in e for k in _ADD_EVENT_KEYWORDS)


def _extract_item_id(payload: dict) -> str | None:
    item = payload.get("Item") or payload.get("item") or {}
    if isinstance(item, dict):
        for k in ("Id", "id", "ItemId", "itemId"):
            if v := item.get(k):
                return str(v)
    for k in ("ItemId", "itemId", "Id", "id"):
        if v := payload.get(k):
            return str(v)
    return None


def _extract_item_type(payload: dict) -> str | None:
    item = payload.get("Item") or payload.get("item") or {}
    if isinstance(item, dict):
        return item.get("Type") or item.get("type")
    return payload.get("Type") or payload.get("type")


@router.post("/webhook/emby")
async def emby_webhook(request: Request) -> dict:
    if settings.webhook_secret:
        token = request.headers.get("X-Babel-Token")
        if token != settings.webhook_secret:
            raise HTTPException(401, "invalid or missing X-Babel-Token")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "request body is not valid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected a JSON object")

    event = str(payload.get("Event") or payload.get("event") or "")
    if event and not _is_added_event(event):
        return {"submitted": False, "reason": f"event not handled: {event!r}"}

    item_type = _extract_item_type(payload)
    if item_type and item_type.lower() not in _VIDEO_TYPES:
        return {"submitted": False, "reason": f"item type not handled: {item_type!r}"}

    item_id = _extract_item_id(payload)
    if not item_id:
        return {"submitted": False, "reason": "no item id found in payload"}

    try:
        emby = emby_client()
        item = emby.get_item(item_id)
        job = submit_item_job(emby=emby, item=item)
    except EmbyError as e:
        raise HTTPException(502, f"Emby lookup failed: {e}") from e
    except ValueError as e:
        return {"submitted": False, "reason": str(e), "item_id": item_id}

    return {"submitted": True, "job_id": job.id, "item_id": item_id, "item_name": item.name}
