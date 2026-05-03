"""Tests for the per-model RTF history that calibrates the openvino
transcribe progress heartbeat to the user's hardware over time."""
from app.pipeline import perf_history


def test_estimated_rtf_returns_default_when_no_history(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_history.settings, "cache_dir", tmp_path, raising=False)
    assert perf_history.estimated_rtf("small", default=0.07) == 0.07
    assert perf_history.estimated_rtf("nonexistent-model", default=0.20) == 0.20


def test_record_then_estimate_uses_measured_rtf(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_history.settings, "cache_dir", tmp_path, raising=False)
    perf_history.record_rtf("small", 0.10)
    perf_history.record_rtf("small", 0.12)
    perf_history.record_rtf("small", 0.11)
    # Median of [0.10, 0.12, 0.11] is 0.11.
    assert perf_history.estimated_rtf("small", default=0.07) == 0.11


def test_history_is_capped_to_max_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_history.settings, "cache_dir", tmp_path, raising=False)
    # Push 15 samples; we keep the last 10. The first 5 should fall off.
    for v in range(15):
        perf_history.record_rtf("medium", 0.05 + v * 0.01)
    # After capping, samples are [v=5..14], i.e. 0.10, 0.11, ..., 0.19.
    # Median is between v=9 and v=10 → 0.145.
    rtf = perf_history.estimated_rtf("medium", default=0.0)
    assert abs(rtf - 0.145) < 1e-9


def test_record_rejects_garbage_values(tmp_path, monkeypatch):
    """An RTF of 0 or negative or absurdly large means we computed it on a
    degenerate audio file. Don't pollute the history."""
    monkeypatch.setattr(perf_history.settings, "cache_dir", tmp_path, raising=False)
    perf_history.record_rtf("small", 0.0)
    perf_history.record_rtf("small", -1.0)
    perf_history.record_rtf("small", 1000.0)
    # No usable samples were stored, so we still get the default.
    assert perf_history.estimated_rtf("small", default=0.07) == 0.07


def test_per_model_isolation(tmp_path, monkeypatch):
    """Recording for whisper-small must not affect the estimate for
    large-v3-turbo. Each model trains its own slope."""
    monkeypatch.setattr(perf_history.settings, "cache_dir", tmp_path, raising=False)
    perf_history.record_rtf("small", 0.07)
    perf_history.record_rtf("large-v3-turbo", 0.18)
    assert perf_history.estimated_rtf("small", default=0.0) == 0.07
    assert perf_history.estimated_rtf("large-v3-turbo", default=0.0) == 0.18


def test_corrupt_history_file_falls_back_to_default(tmp_path, monkeypatch):
    """If someone hand-edits rtf-history.json into garbage, we silently
    fall back to baked-in defaults rather than crashing the pipeline."""
    monkeypatch.setattr(perf_history.settings, "cache_dir", tmp_path, raising=False)
    (tmp_path / "rtf-history.json").write_text("{not valid json")
    assert perf_history.estimated_rtf("small", default=0.07) == 0.07
