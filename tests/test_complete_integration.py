from pathlib import Path

import pytest

import batch_llm.complete as complete_module
from batch_llm.cache import request_cache_hash
from batch_llm.models import BatchJob, BatchResult, BatchStatus


class MemoryStore:
    def ensure_cache_requests(self, requests):
        for request_hash, provider, body in requests:
            self.ensure_cache_request(request_hash, provider, body)

    def cached_responses_many(self, request_hashes):
        return {key: self.cached_responses(key) for key in dict.fromkeys(request_hashes)}

    def __init__(self):
        self.cache = {}

    def ensure_cache_request(self, key, provider, body):
        return None

    def cached_responses(self, key):
        return list(self.cache.get(key, []))

    def add_cached_response(self, key, response, usage=None, *, usage_scope=None, usage_id=None):
        rows = self.cache.setdefault(key, [])
        rows.append(
            BatchResult(
                f"cache-{len(rows)}",
                response,
                usage=usage,
                usage_scope=usage_scope,
                usage_id=usage_id,
            )
        )


class CompleteFakeClient:
    shared_store = MemoryStore()
    submitted = []
    result_factory = None

    def __init__(self, *args, **kwargs):
        self.store = self.shared_store

    def submit(self, prompts, **kwargs):
        prompts = list(prompts)
        self.__class__.submitted.append(prompts)
        return BatchJob(
            id=f"job-{len(self.submitted)}",
            provider="openai",
            model=kwargs["model"],
            status=BatchStatus.SUBMITTED,
            remote_job_id=f"remote-{len(self.submitted)}",
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
        if type(self).result_factory:
            return type(self).result_factory(job, self.submitted[-1])
        return [
            BatchResult(
                f"request-{i}",
                {
                    "choices": [{"message": {"content": f"out:{prompt}"}}],
                    "usage": {"total_tokens": i + 1},
                },
                usage={"total_tokens": i + 1},
                usage_scope="request",
                usage_id=f"{job.id}:request-{i}",
            )
            for i, prompt in enumerate(self.submitted[-1])
        ]


@pytest.fixture(autouse=True)
def reset_fake_client(monkeypatch):
    CompleteFakeClient.shared_store = MemoryStore()
    CompleteFakeClient.submitted = []
    CompleteFakeClient.result_factory = None
    monkeypatch.setattr(complete_module, "BatchClient", CompleteFakeClient)


def test_partial_cache_hit_submits_only_missing_samples(tmp_path: Path):
    body = {"model": "m", "messages": [{"role": "user", "content": "a"}], "temperature": 0}
    key = request_cache_hash("openai", body)
    CompleteFakeClient.shared_store.add_cached_response(
        key,
        {"choices": [{"message": {"content": "cached-a"}}], "usage": {"total_tokens": 3}},
        {"total_tokens": 3},
        usage_scope="request",
        usage_id="old:request-0",
    )

    rows = complete_module.complete_prompts(
        ["a", "b"],
        model="m",
        generation_config={"temperature": 0},
        min_repeat=2,
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )
    assert CompleteFakeClient.submitted == [["a", "b", "b"]]
    assert [(r["prompt"], r["repeat_index"]) for r in rows] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    assert rows[0]["generated_text"] == "cached-a"


def test_second_identical_call_is_fully_cached(tmp_path: Path):
    kwargs = dict(
        model="m",
        generation_config={"temperature": 0},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )
    first = complete_module.complete_prompts(["a", "b"], **kwargs)
    assert len(CompleteFakeClient.submitted) == 1
    second = complete_module.complete_prompts(["a", "b"], **kwargs)
    assert len(CompleteFakeClient.submitted) == 1
    assert [r["generated_text"] for r in first] == [r["generated_text"] for r in second]


def test_same_prompt_changed_config_does_not_use_cache(tmp_path: Path):
    base = dict(model="m", provider="openai", api_key_var="OPENAI_API_KEY", cache_dir=tmp_path)
    complete_module.complete_prompts("a", generation_config={"temperature": 0}, **base)
    complete_module.complete_prompts("a", generation_config={"temperature": 1}, **base)
    assert CompleteFakeClient.submitted == [["a"], ["a"]]


def test_same_prompt_changed_model_does_not_use_cache(tmp_path: Path):
    base = dict(provider="openai", api_key_var="OPENAI_API_KEY", cache_dir=tmp_path)
    complete_module.complete_prompts("a", model="m1", generation_config={}, **base)
    complete_module.complete_prompts("a", model="m2", generation_config={}, **base)
    assert CompleteFakeClient.submitted == [["a"], ["a"]]


def test_provider_result_count_mismatch_raises(tmp_path: Path):
    CompleteFakeClient.result_factory = lambda job, prompts: []
    with pytest.raises(RuntimeError, match="returned 0 results for 1 requests"):
        complete_module.complete_prompts(
            "a",
            model="m",
            generation_config={},
            provider="openai",
            api_key_var="OPENAI_API_KEY",
            cache_dir=tmp_path,
            total_wait_s=0,
        )


def test_completed_batch_retries_until_results_are_available(tmp_path: Path, monkeypatch):
    calls = {"count": 0}

    def delayed_results(job, prompts):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return [
            BatchResult(f"request-{i}", {"choices": [{"message": {"content": f"out:{prompt}"}}]})
            for i, prompt in enumerate(prompts)
        ]

    CompleteFakeClient.result_factory = delayed_results
    monkeypatch.setattr(complete_module.time, "sleep", lambda _seconds: None)
    rows = complete_module.complete_prompts(
        ["a", "b"],
        model="m",
        generation_config={},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
        wait_interval_s=0.001,
        total_wait_s=1,
    )
    assert calls["count"] == 2
    assert [row["generated_text"] for row in rows] == ["out:a", "out:b"]


def test_empty_input_returns_without_client(monkeypatch):
    class ExplodingClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("client should not be constructed")

    monkeypatch.setattr(complete_module, "BatchClient", ExplodingClient)
    assert complete_module.complete_prompts([], model="m", generation_config={}) == []


def test_invalid_provider_and_wrong_key_var_fail_before_api_use(tmp_path: Path):
    with pytest.raises(ValueError, match="provider must"):
        complete_module.complete_prompts(
            "a", model="m", generation_config={}, provider="other", cache_dir=tmp_path
        )
    with pytest.raises(ValueError, match="requires OPENAI_API_KEY"):
        complete_module.complete_prompts(
            "a",
            model="m",
            generation_config={},
            provider="openai",
            api_key_var="GEMINI_API_KEY",
            cache_dir=tmp_path,
        )


def test_overlapping_batches_submit_only_uncached_prompts(tmp_path: Path):
    common = [f"prompt-{i}" for i in range(50, 100)]
    first_prompts = [f"prompt-{i}" for i in range(100)]
    second_prompts = common + [f"prompt-{i}" for i in range(100, 150)]

    kwargs = dict(
        model="m",
        generation_config={"temperature": 0},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    first_rows = complete_module.complete_prompts(first_prompts, **kwargs)
    second_rows = complete_module.complete_prompts(second_prompts, **kwargs)

    assert CompleteFakeClient.submitted[0] == first_prompts
    assert CompleteFakeClient.submitted[1] == [f"prompt-{i}" for i in range(100, 150)]
    assert len(CompleteFakeClient.submitted[1]) == 50

    assert [row["prompt"] for row in first_rows] == first_prompts
    assert [row["prompt"] for row in second_rows] == second_prompts
    assert [row["generated_text"] for row in second_rows] == [
        f"out:{prompt}" for prompt in second_prompts
    ]


def test_duplicate_input_prompts_submit_only_unique_prompts(tmp_path: Path):
    prompts = ["alpha", "beta", "alpha", "gamma", "beta", "alpha"]

    rows = complete_module.complete_prompts(
        prompts,
        model="m",
        generation_config={"temperature": 0},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    assert CompleteFakeClient.submitted == [["alpha", "beta", "gamma"]]
    assert [row["prompt"] for row in rows] == prompts
    assert [row["generated_text"] for row in rows] == [f"out:{prompt}" for prompt in prompts]


def test_output_preserves_all_input_fields_and_adds_generated_text(tmp_path: Path):
    inputs = [
        {
            "id": "sample-a",
            "prompt": "alpha",
            "label": "case-a",
            "score": 1.25,
            "metadata": {"source": "test", "nested": [1, 2, 3]},
            "enabled": True,
        },
        {
            "id": "sample-b",
            "prompt": "beta",
            "label": None,
            "score": 0,
            "metadata": {"source": "other"},
            "enabled": False,
        },
    ]

    rows = complete_module.complete_prompts(
        inputs,
        model="m",
        generation_config={"temperature": 0},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    assert len(rows) == len(inputs)
    for input_row, output_row in zip(inputs, rows):
        for field, value in input_row.items():
            assert field in output_row
            assert output_row[field] == value
        assert output_row["generated_text"] == f"out:{input_row['prompt']}"


@pytest.mark.parametrize(
    ("prompts_factory", "dataset_text_field"),
    [
        (lambda: "alpha", "prompt"),
        (lambda: ["alpha", "beta"], "prompt"),
        (lambda: ("alpha", "beta"), "prompt"),
        (lambda: (prompt for prompt in ["alpha", "beta"]), "prompt"),
        (lambda: [{"id": 1, "prompt": "alpha"}, {"id": 2, "prompt": "beta"}], "prompt"),
        (
            lambda: (row for row in [{"id": 1, "prompt": "alpha"}, {"id": 2, "prompt": "beta"}]),
            "prompt",
        ),
        (lambda: [{"id": 1, "text": "alpha"}, {"id": 2, "text": "beta"}], "text"),
    ],
)
def test_all_supported_input_formats_return_list_of_dicts(
    tmp_path: Path, prompts_factory, dataset_text_field: str
):
    rows = complete_module.complete_prompts(
        prompts_factory(),
        model="m",
        generation_config={"temperature": 0},
        dataset_text_field=dataset_text_field,
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    assert isinstance(rows, list)
    assert rows
    assert all(isinstance(row, dict) for row in rows)
    assert all("generated_text" in row for row in rows)


@pytest.mark.parametrize(
    ("prompts_factory", "expected_prompts"),
    [
        (lambda: "alpha", ["alpha"]),
        (lambda: ["alpha", "beta"], ["alpha", "beta"]),
        (lambda: ("alpha", "beta"), ["alpha", "beta"]),
        (lambda: (prompt for prompt in ["alpha", "beta"]), ["alpha", "beta"]),
    ],
)
def test_string_inputs_return_prompt_field(tmp_path: Path, prompts_factory, expected_prompts):
    rows = complete_module.complete_prompts(
        prompts_factory(),
        model="m",
        generation_config={"temperature": 0},
        provider="openai",
        api_key_var="OPENAI_API_KEY",
        cache_dir=tmp_path,
    )

    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    assert [row["prompt"] for row in rows] == expected_prompts


def test_provider_error_has_readable_message_and_is_logged(tmp_path: Path, caplog):
    def failed_result(job, prompts):
        return [
            BatchResult(
                "request-0",
                None,
                error={
                    "code": "invalid_request_error",
                    "type": "invalid_request_error",
                    "message": "max_tokens is too large",
                },
            )
        ]

    CompleteFakeClient.result_factory = failed_result
    with caplog.at_level("ERROR"):
        rows = complete_module.complete_prompts(
            ["a"],
            model="m",
            generation_config={},
            provider="openai",
            api_key_var="OPENAI_API_KEY",
            cache_dir=tmp_path,
        )

    assert rows[0]["error"] == {
        "code": "invalid_request_error",
        "type": "invalid_request_error",
        "message": "max_tokens is too large",
    }
    assert rows[0]["error_message"] == (
        "code=invalid_request_error, type=invalid_request_error, message=max_tokens is too large"
    )
    assert "request=request-0" in caplog.text
    assert "max_tokens is too large" in caplog.text


def test_nested_provider_error_is_human_readable():
    error = {
        "error": {
            "code": 400,
            "status": "INVALID_ARGUMENT",
            "message": "generation config is invalid",
        }
    }
    assert complete_module._format_provider_error(error) == (
        "code=400, status=INVALID_ARGUMENT, message=generation config is invalid"
    )
