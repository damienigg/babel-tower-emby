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
- Pydantic Field bounds reject obviously-out-of-range numeric settings.

Pre-0.7.32 this file also covered TranslationContext's lazy cinematic
frame plumbing + the scene/cinematic settings bounds. Those went with
the scene/cinematic mode removal.
"""
import json
import time

import pytest

from app.config import _EnvSettings, SettingsStore
from app.jobs import Job, JobCanceled, JobTimeout


# ── Job timeout ──────────────────────────────────────────────────────────────


def test_job_check_cancel_raises_when_deadline_passed():
    j = Job(
        id="x", item_id="i", item_name="n",
        target_lang="fr", provider="nllb",
    )
    j.started_at = time.time() - 10.0
    j.deadline = j.started_at + 1.0   # 9s past the deadline
    with pytest.raises(JobTimeout):
        j.check_cancel()


def test_job_check_cancel_passes_inside_deadline():
    j = Job(
        id="x", item_id="i", item_name="n",
        target_lang="fr", provider="nllb",
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
        target_lang="fr", provider="nllb",
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


def test_pydantic_accepts_zero_job_timeout_means_unlimited(tmp_path):
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}
    s.update({"job_timeout_seconds": 0})
    assert s.job_timeout_seconds == 0
