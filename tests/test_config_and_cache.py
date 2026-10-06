import json
from pathlib import Path

import pytest

from batch_llm.cache import request_cache_hash
from batch_llm.complete import _generated_text, _usage, load_generation_config
from batch_llm.storage import Store, default_cache_dir


def test_load_generation_config_mapping_with_system_prompt():
    config = load_generation_config({"generation": {"temperature": 0}, "system_prompt": "be brief"})
    assert config == {"generation": {"temperature": 0}, "system_prompt": "be brief"}


def test_load_generation_config_rejects_bad_json(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_generation_config(path)


def test_cache_hash_changes_for_arbitrary_generation_option():
    body = {"model": "m", "messages": [{"role": "user", "content": "p"}], "seed": 1}
    assert request_cache_hash("openai", body) != request_cache_hash("openai", {**body, "seed": 2})


def test_cache_hash_provider_is_case_insensitive():
    body = {"model": "m", "messages": [{"role": "user", "content": "p"}]}
    assert request_cache_hash("OPENAI", body) == request_cache_hash("openai", body)


def test_generated_text_openai_and_gemini():
    assert _generated_text({"choices": [{"message": {"content": "hello"}}]}, "openai") == "hello"
    assert (
        _generated_text(
            {"candidates": [{"content": {"parts": [{"text": "hel"}, {"text": "lo"}]}}]}, "gemini"
        )
        == "hello"
    )
    assert _generated_text({}, "openai") is None


def test_usage_preserves_provider_native_payload():
    openai_usage = {"input_tokens": 10, "output_tokens": 2, "details": {"cached_tokens": 3}}
    gemini_usage = {"promptTokenCount": 10, "thoughtsTokenCount": 4}
    assert _usage({"usage": openai_usage}, "openai") == openai_usage
    assert _usage({"usageMetadata": gemini_usage}, "gemini") == gemini_usage


def test_default_cache_dir_prefers_batch_llm_env(monkeypatch, tmp_path: Path):
    override = tmp_path / "explicit"
    monkeypatch.setenv("BATCH_LLM_CACHE_DIR", str(override))
    monkeypatch.setattr("batch_llm.storage.user_cache_dir", lambda _: str(tmp_path / "platform"))
    assert default_cache_dir() == override


def test_default_cache_dir_uses_platformdirs(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("BATCH_LLM_CACHE_DIR", raising=False)
    expected = tmp_path / "xdg" / "batch-llm"
    monkeypatch.setattr("batch_llm.storage.user_cache_dir", lambda _: str(expected))
    assert default_cache_dir() == expected


def test_store_directory_path_creates_state_db(tmp_path: Path):
    store = Store(tmp_path / "cache")
    assert store.path == tmp_path / "cache" / "state.sqlite3"
    assert store.path.exists()


def test_ignored_generation_options_do_not_change_openai_cache_identity():
    from batch_llm.providers.openai import OpenAIProvider

    provider = object.__new__(OpenAIProvider)
    base = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }
    ignored = {
        **base,
        "min_new_tokens": 5,
        "do_sample": False,
        "use_cache": True,
        "truncation": True,
    }

    assert request_cache_hash("openai", provider.cache_identity_body(base)) == request_cache_hash(
        "openai", provider.cache_identity_body(ignored)
    )


def test_translated_generation_option_uses_effective_openai_cache_identity():
    from batch_llm.providers.openai import OpenAIProvider

    provider = object.__new__(OpenAIProvider)
    shared = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]}
    alias = {**shared, "max_new_tokens": 123}
    native = {**shared, "max_completion_tokens": 123}

    assert request_cache_hash("openai", provider.cache_identity_body(alias)) == request_cache_hash(
        "openai", provider.cache_identity_body(native)
    )
