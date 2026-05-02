import json

import pytest

from app.config import SENSITIVE_FIELDS, _EnvSettings, SettingsStore


def _store(tmp_path) -> SettingsStore:
    """A fresh SettingsStore wired to a tmp settings.json. We instantiate a new
    _EnvSettings so its cache_dir matches."""
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}
    return s


def test_proxy_falls_back_to_env_default(tmp_path):
    s = _store(tmp_path)
    # whisper_model defaults to "small" in _EnvSettings
    assert s.whisper_model == "small"


def test_override_takes_precedence_over_env(tmp_path):
    s = _store(tmp_path)
    s.update({"whisper_model": "medium"})
    assert s.whisper_model == "medium"


def test_update_persists_to_file(tmp_path):
    s = _store(tmp_path)
    s.update({"max_line_chars": 50})
    on_disk = json.loads((tmp_path / "settings.json").read_text())
    assert on_disk["max_line_chars"] == 50


def test_update_rejects_unknown_setting(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="Unknown setting"):
        s.update({"this_does_not_exist": 42})


def test_update_validates_value_types(tmp_path):
    s = _store(tmp_path)
    # max_line_chars is int — passing a non-coercible string must fail
    with pytest.raises(ValueError, match="Invalid setting value"):
        s.update({"max_line_chars": "not-a-number"})


def test_reset_drops_override(tmp_path):
    s = _store(tmp_path)
    s.update({"whisper_model": "medium"})
    s.reset("whisper_model")
    assert s.whisper_model == "small"


def test_all_values_masks_sensitive(tmp_path):
    s = _store(tmp_path)
    s.update({"translation_llm_api_key": "sk-real"})
    masked = s.all_values(mask_sensitive=True)
    assert masked["translation_llm_api_key"] == "[set]"

    raw = s.all_values(mask_sensitive=False)
    assert raw["translation_llm_api_key"] == "sk-real"


def test_all_sensitive_fields_are_real_fields():
    """SENSITIVE_FIELDS must reference fields that exist on _EnvSettings."""
    real_fields = set(_EnvSettings.model_fields.keys())
    assert SENSITIVE_FIELDS.issubset(real_fields)


def test_anthropic_api_key_migration(tmp_path):
    """The shared anthropic_api_key was dropped — its value must propagate
    to both per-slot keys when those are blank."""
    legacy = {"anthropic_api_key": "sk-shared"}
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(legacy))

    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)

    assert "anthropic_api_key" not in s._overrides
    assert s._overrides["translation_llm_api_key"] == "sk-shared"
    assert s._overrides["vision_llm_api_key"] == "sk-shared"


def test_anthropic_api_key_migration_does_not_clobber_existing(tmp_path):
    """If a slot already has its own key, the legacy fallback must not overwrite it."""
    legacy = {
        "anthropic_api_key": "sk-shared",
        "translation_llm_api_key": "sk-translation-specific",
    }
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(legacy))

    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)

    assert s._overrides["translation_llm_api_key"] == "sk-translation-specific"
    assert s._overrides["vision_llm_api_key"] == "sk-shared"


def test_legacy_settings_migration(tmp_path):
    """Settings written by an older version with unified `llm_backend` etc.
    must auto-migrate to the per-function slots on load."""
    legacy = {
        "default_translation_provider": "claude",
        "llm_backend": "openai_compat",
        "openai_compat_base_url": "http://ollama:11434/v1",
        "openai_compat_api_key": "secret",
        "openai_compat_model": "qwen2.5:72b",
        "llm_supports_vision": True,
    }
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(legacy))

    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)

    # claude → llm
    assert s._overrides.get("default_translation_provider") == "llm"
    # llm_backend → translation_llm_type + vision_llm_type
    assert s._overrides["translation_llm_type"] == "openai_compat"
    assert s._overrides["vision_llm_type"] == "openai_compat"
    # openai_compat_base_url → both endpoints
    assert s._overrides["translation_llm_endpoint"] == "http://ollama:11434/v1"
    assert s._overrides["vision_llm_endpoint"] == "http://ollama:11434/v1"
    # openai_compat_api_key → both api_keys
    assert s._overrides["translation_llm_api_key"] == "secret"
    assert s._overrides["vision_llm_api_key"] == "secret"
    # openai_compat_model → both models
    assert s._overrides["translation_llm_model"] == "qwen2.5:72b"
    assert s._overrides["vision_llm_model"] == "qwen2.5:72b"
    # llm_supports_vision → vision_llm_enabled + translation_llm_supports_vision
    assert s._overrides["vision_llm_enabled"] is True
    assert s._overrides["translation_llm_supports_vision"] is True
    # legacy keys removed
    assert "llm_backend" not in s._overrides
    assert "openai_compat_base_url" not in s._overrides
