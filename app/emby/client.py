"""Minimal Emby REST client. Used to resolve item paths, list items missing
target-language subs, and trigger metadata refresh after we write a new .vtt.

API auth is via an Emby API key — generate one at Emby Server → Settings →
Advanced → API Keys.
"""
from dataclasses import dataclass
from typing import Iterable

import httpx


@dataclass
class EmbyItem:
    id: str
    name: str
    path: str
    type: str  # "Movie", "Episode", etc.
    media_streams: list[dict]

    def has_subtitle_track(self, lang: str) -> bool:
        lang = lang.lower()
        for s in self.media_streams or []:
            if (s.get("Type") or "").lower() == "subtitle":
                stream_lang = (s.get("Language") or "").lower()
                if stream_lang == lang or stream_lang == _to_iso639_2(lang):
                    return True
        return False


@dataclass
class EmbyPage:
    items: list[EmbyItem]
    total: int       # total count across the whole library, not just this page


_TWO_TO_THREE = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita", "pt": "por",
    "ja": "jpn", "ko": "kor", "zh": "zho", "ru": "rus", "ar": "ara", "hi": "hin",
    "nl": "nld", "sv": "swe", "no": "nor", "da": "dan", "fi": "fin", "pl": "pol",
    "el": "ell", "he": "heb", "tr": "tur", "vi": "vie", "th": "tha", "uk": "ukr",
    "id": "ind", "cs": "ces", "ro": "ron", "hu": "hun", "ca": "cat",
}


def _to_iso639_2(code: str) -> str:
    return _TWO_TO_THREE.get(code, code)


class EmbyError(Exception):
    pass


class EmbyClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        if not base_url or not api_key:
            raise EmbyError("Emby URL and API key are required")
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(
            headers={"X-Emby-Token": api_key, "Accept": "application/json"},
            timeout=30.0,
        )

    def health(self) -> bool:
        try:
            r = self._http.get(f"{self._base}/System/Info/Public")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def get_item(self, item_id: str) -> EmbyItem:
        r = self._http.get(f"{self._base}/Items/{item_id}", params={"Fields": "Path,MediaStreams"})
        if r.status_code != 200:
            raise EmbyError(f"Emby GET /Items/{item_id} → HTTP {r.status_code}: {r.text}")
        d = r.json()
        return EmbyItem(
            id=d["Id"],
            name=d.get("Name") or "",
            path=d.get("Path") or "",
            type=d.get("Type") or "",
            media_streams=d.get("MediaStreams") or [],
        )

    def list_videos(
        self,
        *,
        types: Iterable[str] = ("Movie", "Episode"),
        start_index: int = 0,
        limit: int = 200,
        search_term: str | None = None,
    ) -> EmbyPage:
        """One page of video items + the total count Emby reports for the query.
        For full-library iteration use iter_videos()."""
        params: dict = {
            "Recursive": "true",
            "IncludeItemTypes": ",".join(types),
            "Fields": "Path,MediaStreams",
            "StartIndex": start_index,
            "Limit": limit,
        }
        if search_term:
            params["SearchTerm"] = search_term
        r = self._http.get(f"{self._base}/Items", params=params)
        if r.status_code != 200:
            raise EmbyError(f"Emby GET /Items → HTTP {r.status_code}: {r.text}")
        body = r.json()
        items = [
            EmbyItem(
                id=it["Id"],
                name=it.get("Name") or "",
                path=it.get("Path") or "",
                type=it.get("Type") or "",
                media_streams=it.get("MediaStreams") or [],
            )
            for it in body.get("Items") or []
        ]
        return EmbyPage(items=items, total=int(body.get("TotalRecordCount", len(items))))

    def iter_videos(
        self,
        *,
        types: Iterable[str] = ("Movie", "Episode"),
        page_size: int = 200,
        max_items: int | None = None,
    ):
        """Iterate every video in the library, paging server-side. Stops when a
        page returns fewer items than requested (last page) or when `max_items`
        is hit."""
        types = tuple(types)
        seen = 0
        start = 0
        while True:
            page = self.list_videos(types=types, start_index=start, limit=page_size)
            for it in page.items:
                if max_items is not None and seen >= max_items:
                    return
                yield it
                seen += 1
            if len(page.items) < page_size:
                return
            start += page_size

    def items_missing_subtitle(self, target_lang: str, **kwargs) -> list[EmbyItem]:
        return [it for it in self.list_videos(**kwargs).items if not it.has_subtitle_track(target_lang)]

    def refresh_item(self, item_id: str) -> None:
        r = self._http.post(
            f"{self._base}/Items/{item_id}/Refresh",
            params={"MetadataRefreshMode": "Default", "ImageRefreshMode": "Default"},
        )
        if r.status_code not in (200, 204):
            raise EmbyError(f"Emby POST /Items/{item_id}/Refresh → HTTP {r.status_code}: {r.text}")
