import json
from pathlib import Path

import pytest

from batch_llm.complete import _normalize_inputs, complete_prompts, load_generation_config


def test_load_generation_config_file_resolves_system_prompt(tmp_path: Path):
    (tmp_path / "system.txt").write_text("system text\n", encoding="utf-8")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"generation": {"temperature": 0}, "system_prompt_file": "system.txt"}),
        encoding="utf-8",
    )
    config = load_generation_config(path)
    assert config["generation"] == {"temperature": 0}
    assert config["system_prompt"] == "system text"


def test_load_generation_config_accepts_generation_dict():
    config = load_generation_config({"temperature": 0, "max_tokens": 20})
    assert config == {"generation": {"temperature": 0, "max_tokens": 20}}


def test_normalize_single_string():
    assert _normalize_inputs("hello", "prompt") == [{"prompt": "hello"}]


def test_normalize_strings():
    assert _normalize_inputs(["a", "b"], "prompt") == [{"prompt": "a"}, {"prompt": "b"}]


def test_normalize_mappings_preserves_fields():
    assert _normalize_inputs([{"id": 1, "text": "hello"}], "text") == [{"id": 1, "text": "hello"}]


def test_normalize_requires_string_prompt():
    with pytest.raises(TypeError):
        _normalize_inputs([{"prompt": 123}], "prompt")


def test_complete_prompts_preserves_input_order_duplicates_and_usage(monkeypatch, tmp_path: Path):
    from batch_llm.models import BatchJob, BatchResult, BatchStatus
    import batch_llm.complete as complete_module

    class FakeStore:
        def __init__(self):
            self.rows = {}

        def ensure_cache_requests(self, requests):
            return None

        def cached_responses_many(self, keys):
            return {key: list(self.rows.get(key, [])) for key in keys}

        def add_cached_response(
            self, key, response, usage=None, *, usage_scope=None, usage_id=None
        ):
            from batch_llm.models import BatchResult

            self.rows.setdefault(key, []).append(
                BatchResult(
                    f"cache-{len(self.rows.get(key, []))}",
                    response,
                    usage=usage,
                    usage_scope=usage_scope,
                    usage_id=usage_id,
                )
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.store = FakeStore()

        def submit(self, prompts, **kwargs):
            self.prompts = list(prompts)
            return BatchJob(
                id="job-1",
                provider="openai",
                model="m",
                status=BatchStatus.SUBMITTED,
                remote_job_id="remote-1",
            )

        def wait(self, job, **kwargs):
            return BatchJob(
                id=job.id,
                provider=job.provider,
                model=job.model,
                status=BatchStatus.COMPLETED,
                remote_job_id=job.remote_job_id,
            )

        def results(self, job, refresh=False):
            # Unique prompts are submitted as ["b", "a"], but the output must
            # later be expanded back to source order ["b", "a", "b"].
            return [
                BatchResult(
                    "request-0",
                    {"choices": [{"message": {"content": "B"}}], "usage": {"total_tokens": 11}},
                    usage={"total_tokens": 11},
                    usage_scope="request",
                    usage_id="job-1:request-0",
                ),
                BatchResult(
                    "request-1",
                    {"choices": [{"message": {"content": "A"}}], "usage": {"total_tokens": 7}},
                    usage={"total_tokens": 7},
                    usage_scope="request",
                    usage_id="job-1:request-1",
                ),
            ]

    monkeypatch.setattr(complete_module, "BatchClient", FakeClient)
    rows = complete_module.complete_prompts(
        [{"id": 10, "prompt": "b"}, {"id": 11, "prompt": "a"}, {"id": 12, "prompt": "b"}],
        model="m",
        generation_config={},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    assert [row["id"] for row in rows] == [10, 11, 12]
    assert [row["generated_text"] for row in rows] == ["B", "A", "B"]
    assert [row["usage"]["total_tokens"] for row in rows] == [11, 7, 11]
    assert [row["usage_scope"] for row in rows] == ["request", "request", "request"]
    assert [row["usage_id"] for row in rows] == [
        "job-1:request-0",
        "job-1:request-1",
        "job-1:request-0",
    ]


def test_request_cache_key_includes_provider_model_prompt_and_config():
    from batch_llm.cache import request_cache_hash

    base = {
        "model": "model-a",
        "messages": [
            {"role": "system", "content": "system-a"},
            {"role": "user", "content": "prompt-a"},
        ],
        "temperature": 0,
        "max_tokens": 100,
    }

    key = request_cache_hash("openai", base)

    variants = [
        ("gemini", base),
        ("openai", {**base, "model": "model-b"}),
        (
            "openai",
            {
                **base,
                "messages": [
                    {"role": "system", "content": "system-a"},
                    {"role": "user", "content": "prompt-b"},
                ],
            },
        ),
        (
            "openai",
            {
                **base,
                "messages": [
                    {"role": "system", "content": "system-b"},
                    {"role": "user", "content": "prompt-a"},
                ],
            },
        ),
        ("openai", {**base, "temperature": 0.5}),
        ("openai", {**base, "max_tokens": 200}),
    ]

    for provider, body in variants:
        assert request_cache_hash(provider, body) != key


def test_request_cache_key_is_order_independent_for_config_mapping():
    from batch_llm.cache import request_cache_hash

    body_a = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "prompt-a"}],
        "temperature": 0,
        "max_tokens": 100,
    }
    body_b = {
        "max_tokens": 100,
        "temperature": 0,
        "messages": [{"content": "prompt-a", "role": "user"}],
        "model": "model-a",
    }

    assert request_cache_hash("openai", body_a) == request_cache_hash("openai", body_b)


def test_normalize_missing_text_field():
    with pytest.raises(KeyError, match="missing dataset text field"):
        _normalize_inputs([{"id": 1}], "prompt")


def test_normalize_rejects_non_string_non_mapping():
    with pytest.raises(TypeError, match="must be a string or mapping"):
        _normalize_inputs([123], "prompt")


def test_complete_rejects_invalid_min_repeat():
    import batch_llm.complete as complete_module

    with pytest.raises(ValueError, match="min_repeat"):
        complete_module.complete_prompts("a", model="m", generation_config={}, min_repeat=0)


def test_wait_interval_uses_env_when_not_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("BATCH_LLM_WAIT_INTERVAL_S", "7.5")
    seen = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.store = type(
                "Store",
                (),
                {
                    "ensure_cache_requests": lambda self, *args, **kwargs: None,
                    "cached_responses_many": lambda self, keys: {key: [] for key in keys},
                    "add_cached_response": lambda self, *args, **kwargs: None,
                },
            )()

        def submit(self, *args, **kwargs):
            return type("Job", (), {"id": "job", "completed": False})()

        def wait(self, job, *, poll_interval, timeout):
            seen["poll_interval"] = poll_interval
            job.completed = True
            job.status = type("Status", (), {"value": "completed"})()
            job.error = None
            job.remote_job_id = "remote"
            return job

        def results(self, job, refresh=False):
            return [
                type(
                    "Result",
                    (),
                    {
                        "response": {"choices": [{"message": {"content": "ok"}}]},
                        "error": None,
                        "usage": None,
                        "usage_scope": None,
                        "usage_id": None,
                        "custom_id": "0",
                    },
                )()
            ]

    monkeypatch.setattr("batch_llm.complete.BatchClient", FakeClient)
    complete_prompts("hello", model="gpt-test", generation_config={}, cache_dir=tmp_path)
    assert seen["poll_interval"] == 7.5


def test_explicit_wait_interval_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BATCH_LLM_WAIT_INTERVAL_S", "7.5")
    seen = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.store = type(
                "Store",
                (),
                {
                    "ensure_cache_requests": lambda self, *args, **kwargs: None,
                    "cached_responses_many": lambda self, keys: {key: [] for key in keys},
                    "add_cached_response": lambda self, *args, **kwargs: None,
                },
            )()

        def submit(self, *args, **kwargs):
            return type("Job", (), {"id": "job", "completed": False})()

        def wait(self, job, *, poll_interval, timeout):
            seen["poll_interval"] = poll_interval
            job.completed = True
            job.status = type("Status", (), {"value": "completed"})()
            job.error = None
            job.remote_job_id = "remote"
            return job

        def results(self, job, refresh=False):
            return [
                type(
                    "Result",
                    (),
                    {
                        "response": {"choices": [{"message": {"content": "ok"}}]},
                        "error": None,
                        "usage": None,
                        "usage_scope": None,
                        "usage_id": None,
                        "custom_id": "0",
                    },
                )()
            ]

    monkeypatch.setattr("batch_llm.complete.BatchClient", FakeClient)
    complete_prompts(
        "hello", model="gpt-test", generation_config={}, wait_interval_s=2, cache_dir=tmp_path
    )
    assert seen["poll_interval"] == 2


def test_invalid_wait_interval_env(monkeypatch):
    monkeypatch.setenv("BATCH_LLM_WAIT_INTERVAL_S", "nope")
    with pytest.raises(ValueError, match="BATCH_LLM_WAIT_INTERVAL_S must be a number"):
        complete_prompts("hello", model="gpt-test", generation_config={})


def test_load_generation_config_mapping_with_absolute_system_prompt_file(tmp_path: Path):
    prompt = tmp_path / "system.md"
    prompt.write_text("absolute system\n", encoding="utf-8")

    config = load_generation_config(
        {"generation": {"temperature": 0}, "system_prompt_file": str(prompt)}
    )

    assert config["generation"] == {"temperature": 0}
    assert config["system_prompt"] == "absolute system"
    assert config["system_prompt_file"] == str(prompt.resolve())


def test_load_generation_config_file_respects_absolute_system_prompt_file(tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt = prompt_dir / "system.md"
    prompt.write_text("absolute from config\n", encoding="utf-8")

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "generation.json"
    config_path.write_text(
        json.dumps({"generation": {"temperature": 0}, "system_prompt_file": str(prompt.resolve())}),
        encoding="utf-8",
    )

    config = load_generation_config(config_path)

    assert config["generation"] == {"temperature": 0}
    assert config["system_prompt"] == "absolute from config"
    assert config["system_prompt_file"] == str(prompt.resolve())


def test_load_generation_config_mapping_relative_system_prompt_file_rejected():
    with pytest.raises(
        ValueError,
        match="system_prompt_file must be an absolute path when generation_config is provided as a mapping",
    ):
        load_generation_config(
            {"generation": {"temperature": 0}, "system_prompt_file": "system.md"}
        )


def test_load_generation_config_shorthand_with_system_prompt():
    config = load_generation_config(
        {"system_prompt": "be brief", "temperature": 0, "max_tokens": 20}
    )

    assert config == {
        "system_prompt": "be brief",
        "generation": {"temperature": 0, "max_tokens": 20},
    }


def test_load_generation_config_shorthand_with_system_prompt_file(tmp_path: Path):
    prompt = tmp_path / "system.md"
    prompt.write_text("be brief\n", encoding="utf-8")

    config = load_generation_config({"system_prompt_file": str(prompt), "temperature": 0})

    assert config["generation"] == {"temperature": 0}
    assert config["system_prompt"] == "be brief"


def test_load_generation_config_rejects_system_prompt_and_file(tmp_path: Path):
    prompt = tmp_path / "system.md"
    prompt.write_text("file prompt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_generation_config(
            {"generation": {}, "system_prompt": "inline prompt", "system_prompt_file": str(prompt)}
        )


def test_load_generation_config_merges_nested_and_top_level_generation_options():
    config = load_generation_config(
        {
            "generation": {"temperature": 0},
            "system_prompt": "be brief",
            "max_tokens": 20,
            "min_new_tokens": 5,
        }
    )

    assert config == {
        "system_prompt": "be brief",
        "generation": {"temperature": 0, "max_tokens": 20, "min_new_tokens": 5},
    }


def test_load_generation_config_rejects_duplicate_generation_option_locations():
    with pytest.raises(ValueError, match="specified both at top level and inside 'generation'"):
        load_generation_config({"generation": {"temperature": 0}, "temperature": 1})


def test_system_prompt_cache_identity_uses_resolved_content_not_file_path(tmp_path: Path):
    from batch_llm.cache import request_cache_hash

    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("same system prompt\n", encoding="utf-8")
    second.write_text("same system prompt\n", encoding="utf-8")

    cfg_a = load_generation_config({"system_prompt_file": str(first.resolve()), "temperature": 0})
    cfg_b = load_generation_config({"system_prompt_file": str(second.resolve()), "temperature": 0})

    def body(cfg):
        return {
            "model": "model-a",
            "messages": [
                {"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": "prompt-a"},
            ],
            **cfg["generation"],
        }

    key_a = request_cache_hash("openai", body(cfg_a))
    key_b = request_cache_hash("openai", body(cfg_b))
    assert key_a == key_b

    second.write_text("different system prompt\n", encoding="utf-8")
    cfg_c = load_generation_config({"system_prompt_file": str(second.resolve()), "temperature": 0})
    assert request_cache_hash("openai", body(cfg_c)) != key_a


def test_batch_job_hash_uses_resolved_system_prompt_not_file_path(tmp_path: Path):
    from batch_llm.client import BatchClient

    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("same system prompt\n", encoding="utf-8")
    second.write_text("same system prompt\n", encoding="utf-8")

    cfg_a = load_generation_config({"system_prompt_file": str(first.resolve())})
    cfg_b = load_generation_config({"system_prompt_file": str(second.resolve())})

    def requests(cfg):
        return [
            (
                "request-0",
                {
                    "model": "model-a",
                    "messages": [
                        {"role": "system", "content": cfg["system_prompt"]},
                        {"role": "user", "content": "prompt-a"},
                    ],
                },
            )
        ]

    assert BatchClient._hash_request(
        "openai", "model-a", requests(cfg_a)
    ) == BatchClient._hash_request("openai", "model-a", requests(cfg_b))


def test_complete_prompts_warns_for_ignored_generation_options(monkeypatch, tmp_path):
    class FakeStore:
        def ensure_cache_requests(self, *args, **kwargs):
            return None

        def cached_responses_many(self, keys):
            return {key: [] for key in keys}

        def add_cached_response(self, *args, **kwargs):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.store = FakeStore()

        def submit(self, prompts, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(
                id="job",
                remote_job_id="remote",
                completed=True,
                status=SimpleNamespace(value="completed"),
                error=None,
            )

        def wait(self, job, **kwargs):
            return job

        def results(self, job, refresh=False):
            from batch_llm.models import BatchResult

            return [
                BatchResult(
                    custom_id="request-0", response={"choices": [{"message": {"content": "ok"}}]}
                )
            ]

    monkeypatch.setattr("batch_llm.complete.BatchClient", FakeClient)
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    with pytest.warns(UserWarning, match=r"min_new_tokens.*use_cache"):
        complete_prompts(
            ["hello"],
            model="gpt-test",
            generation_config={"temperature": 0, "min_new_tokens": 1, "use_cache": True},
            cache_dir=tmp_path,
            wait_interval_s=0.001,
            total_wait_s=1,
        )


def test_complete_prompts_fully_cached_uses_one_bulk_read_and_no_cache_write(monkeypatch, tmp_path):
    from batch_llm.models import BatchResult
    import batch_llm.complete as complete_module

    calls = {"lookup": 0, "ensure": 0}

    class FakeStore:
        def cached_responses_many(self, keys):
            calls["lookup"] += 1
            return {
                key: [BatchResult("cache-0", {"choices": [{"message": {"content": "cached"}}]})]
                for key in keys
            }

        def ensure_cache_requests(self, requests):
            calls["ensure"] += 1

        def add_cached_response(self, *args, **kwargs):
            raise AssertionError("fully cached call must not add responses")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.store = FakeStore()

        def submit(self, *args, **kwargs):
            raise AssertionError("fully cached call must not submit")

    monkeypatch.setattr(complete_module, "BatchClient", FakeClient)
    rows = complete_module.complete_prompts(
        ["a", "b", "c"],
        model="gpt-test",
        generation_config={},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    assert len(rows) == 3
    assert calls == {"lookup": 1, "ensure": 0}


def test_complete_prompts_only_persists_missing_cache_requests(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from batch_llm.models import BatchResult
    import batch_llm.complete as complete_module

    ensured = []

    class FakeStore:
        def cached_responses_many(self, keys):
            return {
                keys[0]: [
                    BatchResult("cache-0", {"choices": [{"message": {"content": "cached"}}]})
                ],
                keys[1]: [],
            }

        def ensure_cache_requests(self, requests):
            ensured.extend(requests)

        def add_cached_response(self, *args, **kwargs):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.store = FakeStore()

        def submit(self, prompts, **kwargs):
            assert list(prompts) == ["missing"]
            return SimpleNamespace(
                id="job",
                remote_job_id="remote",
                completed=True,
                status=SimpleNamespace(value="completed"),
                error=None,
            )

        def wait(self, job, **kwargs):
            return job

        def results(self, job, refresh=False):
            return [BatchResult("request-0", {"choices": [{"message": {"content": "new"}}]})]

    monkeypatch.setattr(complete_module, "BatchClient", FakeClient)
    rows = complete_module.complete_prompts(
        ["cached", "missing"],
        model="gpt-test",
        generation_config={},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    assert [row["generated_text"] for row in rows] == ["cached", "new"]
    assert len(ensured) == 1
    assert ensured[0][1] == "openai"
