import json

import pytest

from batch_llm import complete
from batch_llm.models import BatchJob, BatchResult, BatchStatus


def _cached_result(text: str) -> BatchResult:
    return BatchResult(custom_id="cached", response={"choices": [{"message": {"content": text}}]})


def test_complete_prompts_fully_cached_uses_one_bulk_read_and_no_cache_write(monkeypatch, tmp_path):
    import batch_llm.complete as complete_module

    calls = {"lookup": 0, "ensure": 0, "submit": 0}

    class FakeStore:
        def cached_responses_many(self, keys):
            calls["lookup"] += 1
            return {key: [_cached_result("cached")] for key in keys}

        def ensure_cache_requests(self, rows):
            calls["ensure"] += 1

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.store = FakeStore()

        def submit(self, *args, **kwargs):
            calls["submit"] += 1
            raise AssertionError("fully cached prompts must not be submitted")

    monkeypatch.setattr(complete_module, "BatchClient", FakeClient)

    rows = complete_module.complete_prompts(
        ["a", "b"],
        model="m",
        generation_config={},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    assert calls == {"lookup": 1, "ensure": 0, "submit": 0}
    assert [row["generated_text"] for row in rows] == ["cached", "cached"]


def test_complete_prompts_partial_cache_reuses_initial_bulk_read(monkeypatch, tmp_path):
    import batch_llm.complete as complete_module

    calls = {"lookup": 0, "ensure": 0, "submit": 0}
    cached_key = None

    class FakeStore:
        def cached_responses_many(self, keys):
            nonlocal cached_key
            calls["lookup"] += 1
            cached_key = keys[0]
            return {keys[0]: [_cached_result("cached")], keys[1]: []}

        def ensure_cache_requests(self, rows):
            calls["ensure"] += 1
            assert len(rows) == 1

        def add_cached_response(self, *args, **kwargs):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.store = FakeStore()

        def submit(self, prompts, **kwargs):
            calls["submit"] += 1
            assert list(prompts) == ["b"]
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
            return [
                BatchResult(
                    custom_id="request-0", response={"choices": [{"message": {"content": "new"}}]}
                )
            ]

    monkeypatch.setattr(complete_module, "BatchClient", FakeClient)

    rows = complete_module.complete_prompts(
        ["a", "b"],
        model="m",
        generation_config={},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
        wait_interval_s=0.001,
    )

    assert cached_key is not None
    assert calls == {"lookup": 1, "ensure": 1, "submit": 1}
    assert [row["generated_text"] for row in rows] == ["cached", "new"]


def test_reconcile_submission_returns_jobs_that_do_not_need_recovery():
    from types import SimpleNamespace

    from batch_llm.client import BatchClient
    from batch_llm.models import BatchStatus

    client = object.__new__(BatchClient)

    with_remote_id = SimpleNamespace(remote_job_id="batch-1", status=BatchStatus.SUBMITTING)
    assert client._reconcile_submission(with_remote_id) is with_remote_id

    already_submitted = SimpleNamespace(remote_job_id=None, status=BatchStatus.SUBMITTED)
    assert client._reconcile_submission(already_submitted) is already_submitted


def test_reconcile_submission_recovers_by_provider_file_id():
    from types import SimpleNamespace

    from batch_llm.client import BatchClient
    from batch_llm.models import BatchStatus

    job = SimpleNamespace(
        id="local-job", remote_job_id=None, remote_file_id="file-1", status=BatchStatus.SUBMITTING
    )

    class Store:
        def __init__(self):
            self.updates = []

        def update_job(self, job_id, **fields):
            self.updates.append((job_id, fields))
            for key, value in fields.items():
                setattr(job, key, value)

        def get_job(self, job_id):
            assert job_id == job.id
            return job

    provider = SimpleNamespace(
        name="openai",
        find_created_job=lambda *, job_id: None,
        find_created_job_by_file_id=lambda *, file_id: (
            "batch-recovered" if file_id == "file-1" else None
        ),
    )

    client = object.__new__(BatchClient)
    client.provider = provider
    client.store = Store()

    recovered = client._reconcile_submission(job)

    assert recovered.remote_job_id == "batch-recovered"
    assert recovered.status == BatchStatus.SUBMITTED
    assert client.store.updates == [
        ("local-job", {"remote_job_id": "batch-recovered", "status": BatchStatus.SUBMITTED})
    ]


def test_find_openai_job_by_input_file_paginates():
    from types import SimpleNamespace

    from batch_llm.client import BatchClient

    class Page:
        def __init__(self, data, next_page=None):
            self.data = data
            self.next_page = next_page

        def has_next_page(self):
            return self.next_page is not None

        def get_next_page(self):
            return self.next_page

    second = Page(
        [
            SimpleNamespace(id="batch-match", input_file_id="file-target"),
            # Duplicate batch IDs should not create an ambiguous match.
            SimpleNamespace(id="batch-match", input_file_id="file-target"),
        ]
    )
    first = Page([SimpleNamespace(id="batch-other", input_file_id="file-other")], next_page=second)

    batches = SimpleNamespace(list=lambda *, limit: first)
    provider = SimpleNamespace(name="openai", client=SimpleNamespace(batches=batches))

    client = object.__new__(BatchClient)
    client.provider = provider

    assert client._find_openai_job_by_input_file("file-target") == "batch-match"


def test_find_openai_job_by_input_file_handles_missing_and_ambiguous_matches():
    from types import SimpleNamespace

    from batch_llm.client import BatchClient, SubmissionUncertainError

    class Page:
        def __init__(self, data):
            self.data = data

        def has_next_page(self):
            return False

        def get_next_page(self):
            raise AssertionError("should not request another page")

    client = object.__new__(BatchClient)

    client.provider = SimpleNamespace(
        name="openai",
        client=SimpleNamespace(
            batches=SimpleNamespace(
                list=lambda *, limit: Page(
                    [SimpleNamespace(id="other", input_file_id="other-file")]
                )
            )
        ),
    )
    assert client._find_openai_job_by_input_file("target") is None

    client.provider = SimpleNamespace(
        name="openai",
        client=SimpleNamespace(
            batches=SimpleNamespace(
                list=lambda *, limit: Page(
                    [
                        SimpleNamespace(id="batch-1", input_file_id="target"),
                        SimpleNamespace(id="batch-2", input_file_id="target"),
                    ]
                )
            )
        ),
    )
    with pytest.raises(SubmissionUncertainError, match="multiple OpenAI batches"):
        client._find_openai_job_by_input_file("target")


def test_find_openai_job_by_input_file_handles_unavailable_client():
    from types import SimpleNamespace

    from batch_llm.client import BatchClient

    client = object.__new__(BatchClient)

    client.provider = SimpleNamespace(name="openai")
    assert client._find_openai_job_by_input_file("file-1") is None

    client.provider = SimpleNamespace(
        name="openai", client=SimpleNamespace(batches=SimpleNamespace())
    )
    assert client._find_openai_job_by_input_file("file-1") is None


def test_load_generation_config_merges_nested_and_top_level_generation():
    config = complete.load_generation_config(
        {"generation": {"temperature": 0}, "max_tokens": 100, "system_prompt": "system"}
    )

    assert config["generation"] == {"temperature": 0, "max_tokens": 100}
    assert config["system_prompt"] == "system"


def test_load_generation_config_rejects_duplicate_generation_options():
    with pytest.raises(ValueError, match="specified both"):
        complete.load_generation_config({"generation": {"temperature": 0}, "temperature": 1})


def test_load_generation_config_rejects_invalid_reserved_values(tmp_path):
    with pytest.raises(TypeError, match="generation must be a mapping"):
        complete.load_generation_config({"generation": "invalid"})

    with pytest.raises(
        ValueError, match="system_prompt and system_prompt_file are mutually exclusive"
    ):
        complete.load_generation_config(
            {"system_prompt": "hello", "system_prompt_file": tmp_path / "prompt.md"}
        )

    with pytest.raises(TypeError, match="system_prompt must be a string"):
        complete.load_generation_config({"system_prompt": 123})

    with pytest.raises(TypeError, match="system_prompt_file must be a path"):
        complete.load_generation_config({"system_prompt_file": 123})

    with pytest.raises(ValueError, match="must be an absolute path"):
        complete.load_generation_config({"system_prompt_file": "prompt.md"})


def test_load_generation_config_reads_relative_system_prompt_file(tmp_path):
    prompt_file = tmp_path / "system.md"
    prompt_file.write_text("  system prompt\n", encoding="utf-8")

    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"generation": {"temperature": 0}, "system_prompt_file": "system.md"}),
        encoding="utf-8",
    )

    config = complete.load_generation_config(config_file)

    assert config["system_prompt"] == "system prompt"
    assert config["system_prompt_file"] == str(prompt_file.resolve())


def test_normalize_inputs_validation():
    assert complete._normalize_inputs("hello", "prompt") == [{"prompt": "hello"}]

    assert complete._normalize_inputs([{"prompt": "hello", "id": 1}], "prompt") == [
        {"prompt": "hello", "id": 1}
    ]

    with pytest.raises(KeyError, match="missing dataset text field"):
        complete._normalize_inputs([{"text": "hello"}], "prompt")

    with pytest.raises(TypeError, match="must be a string or mapping"):
        complete._normalize_inputs([123], "prompt")

    with pytest.raises(TypeError, match="must be a string"):
        complete._normalize_inputs([{"prompt": 123}], "prompt")


def test_generated_text_supports_openai_and_gemini_and_invalid_responses():
    assert (
        complete._generated_text({"choices": [{"message": {"content": "openai text"}}]}, "openai")
        == "openai text"
    )

    assert (
        complete._generated_text(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "gemini "}, {"text": "text"}, "ignored"]}}
                ]
            },
            "gemini",
        )
        == "gemini text"
    )

    assert complete._generated_text(None, "openai") is None
    assert complete._generated_text({"unexpected": True}, "openai") is None


def test_usage_handles_supported_shapes_and_invalid_values():
    assert complete._usage(None, "openai") is None

    assert complete._usage({"usage": {"total_tokens": 10}}, "openai") == {"total_tokens": 10}

    assert complete._usage({"usageMetadata": {"totalTokenCount": 10}}, "gemini") == {
        "totalTokenCount": 10
    }

    assert complete._usage({"usage_metadata": {"total_tokens": 10}}, "openai") == {
        "total_tokens": 10
    }

    assert complete._usage({"usage": "invalid"}, "openai") is None


def test_format_provider_error_variants():
    assert complete._format_provider_error(None) == ""
    assert complete._format_provider_error("failure") == "failure"

    assert (
        complete._format_provider_error({"error": {"code": "bad", "message": "failure"}})
        == "code=bad, message=failure"
    )

    assert complete._format_provider_error(
        {"code": 400, "type": "invalid_request", "status": "FAILED", "message": "bad request"}
    ) == ("code=400, type=invalid_request, status=FAILED, message=bad request")

    assert complete._format_provider_error({"unexpected": "value"}) == ('{"unexpected": "value"}')

    assert complete._format_provider_error(123) == "123"


def test_complete_prompts_rejects_invalid_basic_arguments():
    with pytest.raises(ValueError, match="min_repeat must be >= 1"):
        complete.complete_prompts(["hello"], model="gpt-test", generation_config={}, min_repeat=0)

    assert complete.complete_prompts([], model="gpt-test", generation_config={}) == []

    with pytest.raises(ValueError, match="provider must be"):
        complete.complete_prompts(
            ["hello"], model="gpt-test", generation_config={}, provider="invalid"
        )

    with pytest.raises(ValueError, match="requires OPENAI_API_KEY"):
        complete.complete_prompts(
            ["hello"],
            model="gpt-test",
            generation_config={},
            provider="openai",
            api_key_var="GEMINI_API_KEY",
        )
