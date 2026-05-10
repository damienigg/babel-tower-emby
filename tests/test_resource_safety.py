"""Regression coverage for the resource-safety changes that were introduced
to prevent another TrueNAS-style host OOM. These tests don't exercise the
heavy backends (Whisper, ffmpeg) — they pin down the in-process behaviors
that were the root causes of the crash, so a future refactor can't quietly
undo them.

Covered:
- Job wall-clock deadline (job_timeout_seconds) raises JobTimeout at
  check_cancel even when the user never clicked Cancel.
- Atomic settings.json write: settings persistence uses a temp-file +
  os.replace, and a corrupt file on disk is backed up rather than silently
  dropped.
- TranslationContext lazy cinematic frames: cue_frames_provider is invoked
  on demand per cue id, the cue_ids_with_frames whitelist caps which cues
  call out, and the eager cue_frames dict still wins when both are set
  (so existing test fixtures keep working).
- Pydantic Field bounds reject obviously-out-of-range numeric settings.
"""
import json
import time

import pytest

from app.config import _EnvSettings, SettingsStore
from app.jobs import Job, JobCanceled, JobTimeout
from app.pipeline.translate.base import (
    SceneInfo, TranslationContext,
)


# ── Job timeout ──────────────────────────────────────────────────────────────


def test_job_check_cancel_raises_when_deadline_passed():
    j = Job(
        id="x", item_id="i", item_name="n",
        target_lang="fr", provider="nllb", mode="audio",
    )
    j.started_at = time.time() - 10.0
    j.deadline = j.started_at + 1.0   # 9s past the deadline
    with pytest.raises(JobTimeout):
        j.check_cancel()


def test_job_check_cancel_passes_inside_deadline():
    j = Job(
        id="x", item_id="i", item_name="n",
        target_lang="fr", provider="nllb", mode="audio",
    )
    j.started_at = time.time()
    j.deadline = j.started_at + 3600.0   # plenty of headroom
    # No raise = pass.
    j.check_cancel()


def test_job_timeout_subclasses_canceled_for_compat():
    """Existing exception handlers across the pipeline (e.g. the runner)
    catch JobCanceled. JobTimeout must be a subclass so timeout flows
    through the same code paths until the runner specifically branches on it."""
    assert issubclass(JobTimeout, JobCanceled)


def test_job_check_cancel_user_cancel_wins_over_deadline_check():
    """User cancel should raise JobCanceled (not JobTimeout) even if the
    deadline also happens to have lapsed. Order of checks: user first."""
    j = Job(
        id="x", item_id="i", item_name="n",
        target_lang="fr", provider="nllb", mode="audio",
    )
    j.started_at = time.time() - 10.0
    j.deadline = j.started_at + 1.0
    j.cancel_requested = True
    with pytest.raises(JobCanceled) as ei:
        j.check_cancel()
    # JobTimeout would also pass `isinstance JobCanceled`, so distinguish
    # by the exact type — the user-cancel message wins.
    assert type(ei.value) is JobCanceled


# ── Atomic settings persistence ─────────────────────────────────────────────


def test_settings_save_is_atomic(tmp_path):
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}

    s.update({"max_line_chars": 50})
    # After a successful save, the .tmp sidecar must NOT linger — os.replace
    # is the contract that guarantees this.
    assert not (tmp_path / "settings.json.tmp").exists()
    assert (tmp_path / "settings.json").exists()
    assert json.loads((tmp_path / "settings.json").read_text())["max_line_chars"] == 50


def test_corrupt_settings_file_is_backed_up_not_silently_lost(tmp_path, caplog):
    """When settings.json is unreadable the previous behavior silently
    returned {} (lost every API key the user had configured). New behavior:
    move the corrupt file aside under a timestamped backup name and log."""
    bad = tmp_path / "settings.json"
    bad.write_text("{this is not json")

    env = _EnvSettings()
    env.cache_dir = tmp_path

    with caplog.at_level("WARNING", logger="subtitle_this"):
        store = SettingsStore(env)

    # New store starts empty — but we kept the corrupt content for forensics.
    assert store._overrides == {}
    backups = list(tmp_path.glob("settings.json.corrupt.*"))
    assert len(backups) == 1, f"expected exactly one corrupt backup, got {backups}"
    assert backups[0].read_text() == "{this is not json"
    # And we logged a warning the operator can see in `docker logs`.
    assert any("unreadable" in r.message for r in caplog.records)


# ── Pydantic Field bounds on numeric settings ───────────────────────────────


def test_pydantic_rejects_negative_translation_batch_size(tmp_path):
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}
    with pytest.raises(ValueError, match="Invalid setting value"):
        s.update({"translation_batch_size": -1})


def test_pydantic_rejects_huge_scene_max_scenes(tmp_path):
    """Without bounds, a user could type 9999999 in the UI and OOM the
    scene-bible stage. Hard upper limit at 2000."""
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}
    with pytest.raises(ValueError, match="Invalid setting value"):
        s.update({"scene_max_scenes": 9_999_999})


def test_pydantic_accepts_zero_job_timeout_means_unlimited(tmp_path):
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}
    s.update({"job_timeout_seconds": 0})
    assert s.job_timeout_seconds == 0


def test_pydantic_accepts_zero_cinematic_frames(tmp_path):
    """0 must be a valid value — disables per-cue frame extraction
    entirely (downgrades cinematic to scene-mode behavior)."""
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}
    s.update({"cinematic_max_cues_with_frames": 0})
    assert s.cinematic_max_cues_with_frames == 0


# ── TranslationContext lazy frame plumbing ──────────────────────────────────


def test_translation_context_frame_for_prefers_eager_dict():
    """If a test injects fake JPEGs via cue_frames, those win over any
    provider — so existing test_translate_llm.py fixtures keep working."""
    calls = []
    def provider(cue_id: int) -> bytes:
        calls.append(cue_id)
        return b"from-provider"

    ctx = TranslationContext(
        cue_frames={0: b"eager-bytes"},
        cue_frames_provider=provider,
        cue_ids_with_frames={0, 1},
    )
    assert ctx.frame_for(0) == b"eager-bytes"
    # Eager hit → provider was NOT consulted.
    assert calls == []


def test_translation_context_frame_for_falls_back_to_provider():
    seen = []
    def provider(cue_id: int) -> bytes:
        seen.append(cue_id)
        return b"jpeg-bytes"

    ctx = TranslationContext(
        cue_frames={},
        cue_frames_provider=provider,
        cue_ids_with_frames={5, 6},
    )
    assert ctx.frame_for(5) == b"jpeg-bytes"
    assert seen == [5]


def test_translation_context_frame_for_respects_whitelist():
    """Cues outside the whitelist must NOT trigger the provider — this is
    how the cinematic_max_cues_with_frames cap actually shrinks RAM usage."""
    seen = []
    def provider(cue_id: int) -> bytes | None:
        seen.append(cue_id)
        return b"jpeg"

    ctx = TranslationContext(
        cue_frames_provider=provider,
        cue_ids_with_frames={1},   # cue 2 is NOT in the whitelist
    )
    assert ctx.frame_for(2) is None
    # Provider must not have been called for cue 2 — that's the whole point.
    assert seen == []


def test_translation_context_has_cue_frames_lazy_path():
    """has_cue_frames must report True for the lazy path so the LLM
    provider picks the cinematic batch size."""
    ctx = TranslationContext(
        cue_frames_provider=lambda _: b"x",
        cue_ids_with_frames={0},
    )
    assert ctx.has_cue_frames() is True


def test_translation_context_has_cue_frames_lazy_path_empty_whitelist():
    """If the whitelist is empty (cap = 0), there's no cinematic work to
    do — keep using the text-only batch size."""
    ctx = TranslationContext(
        cue_frames_provider=lambda _: b"x",
        cue_ids_with_frames=set(),
    )
    assert ctx.has_cue_frames() is False


def test_translation_context_has_cue_frames_eager_dict():
    """The eager-dict path stays True so existing scene/cinematic tests
    that build cue_frames manually keep working unchanged."""
    ctx = TranslationContext(cue_frames={0: b"x"})
    assert ctx.has_cue_frames() is True


def test_translation_context_default_is_text_only():
    """Empty context → text-only translation. Used by audio mode."""
    ctx = TranslationContext()
    assert ctx.has_cue_frames() is False
    assert ctx.frame_for(0) is None
