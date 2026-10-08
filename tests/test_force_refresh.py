from pathlib import Path

import pytest

import batch_llm.complete as complete_module
from batch_llm.cache import request_cache_hash
from batch_llm.client import BatchClient
from batch_llm.models import BatchResult, BatchStatus
from batch_llm.providers.base import Provider
from batch_llm.request_identity import effective_request_body
from batch_llm.storage import Store


class RefreshProvider(Provider):
    def __init__(self, name):
        self.name = name
        self.requests = {}
        self.created = []
        self.fail_indices = set()
        self.batch_status = BatchStatus.COMPLETED

    def write_input(self, requests, path):
        self.requests[str(path)] = requests
        path.write_text("input")

    def upload(self, path):
        return str(path)

    def create(self, *, job_id, model, file_id):
        self.created.append(job_id)
        self.requests[job_id] = self.requests[file_id]
        return job_id

    def find_created_job(self, *, job_id):
        return job_id if job_id in self.created else None

    def status(self, remote_job_id):
        return self.batch_status, None

    def results(self, remote_job_id):
        results = []
        for i, (custom_id, body) in enumerate(self.requests[remote_job_id]):
            if i in self.fail_indices:
                results.append(BatchResult(custom_id, None, error="request failed"))
                continue
            text = f"{remote_job_id}:{i}:{body['messages'][-1]['content']}"
            if self.name == "openai":
                response = {"choices": [{"message": {"content": text}}]}
            else:
                response = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
            results.append(
                BatchResult(
                    custom_id, response, usage={"total_tokens": i + 1}, usage_scope="request"
                )
            )
        return results


@pytest.fixture(params=["openai", "gemini"])
def refresh_client(request, monkeypatch, tmp_path):
    provider = RefreshProvider(request.param)
    client = BatchClient(provider, storage_path=tmp_path)
    monkeypatch.setattr(complete_module, "BatchClient", lambda *args, **kwargs: client)
    kwargs = dict(model="m", generation_config={}, provider=provider.name, cache_dir=tmp_path)
    return provider, client, kwargs


def test_force_refresh_replaces_cache_and_submits_fresh_jobs(refresh_client):
    provider, client, kwargs = refresh_client
    prompts = [{"id": 1, "prompt": "a"}, {"id": 2, "prompt": "b"}, {"id": 3, "prompt": "a"}]
    original = complete_module.complete_prompts(prompts, min_repeat=3, **kwargs)
    unrelated = complete_module.complete_prompts("unrelated", **kwargs)

    def reject_cache_read(*args, **kwargs):
        raise AssertionError("force refresh must bypass cache reads")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(client.store, "cached_responses_many", reject_cache_read)
        refreshed = complete_module.complete_prompts(
            prompts, min_repeat=2, force_refresh=True, **kwargs
        )

    assert len(provider.created) == 3
    assert provider.created[0] != provider.created[-1]
    requests = provider.requests[provider.created[-1]]
    assert [body["messages"][-1]["content"] for _, body in requests] == ["a", "a", "b", "b"]
    assert all(set(body) == {"model", "messages"} for _, body in requests)
    assert [(row["id"], row["repeat_index"]) for row in refreshed] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]
    assert refreshed[0]["generated_text"] != original[0]["generated_text"]
    assert refreshed[0]["usage_id"] != original[0]["usage_id"]
    assert refreshed[0]["usage_id"] == refreshed[4]["usage_id"]

    for prompt in ["a", "b"]:
        key = request_cache_hash(
            provider.name,
            effective_request_body(
                provider.name, {"model": "m", "messages": [{"role": "user", "content": prompt}]}
            ),
        )
        assert len(client.store.cached_responses(key)) == 2
    cached = complete_module.complete_prompts(prompts, min_repeat=2, **kwargs)
    for field in ["generated_text", "usage", "usage_scope", "usage_id"]:
        assert [row[field] for row in cached] == [row[field] for row in refreshed]
    assert (
        complete_module.complete_prompts("unrelated", **kwargs)[0]["generated_text"]
        == unrelated[0]["generated_text"]
    )
    assert len(provider.created) == 3

    again = complete_module.complete_prompts(prompts, min_repeat=2, force_refresh=True, **kwargs)
    assert len(provider.created) == 4
    assert again[0]["batch_job_id"] != refreshed[0]["batch_job_id"]
    assert again[0]["generated_text"] != refreshed[0]["generated_text"]


def test_force_refresh_without_cached_responses(refresh_client):
    provider, _, kwargs = refresh_client
    rows = complete_module.complete_prompts("a", force_refresh=True, **kwargs)
    assert len(provider.created) == 1
    assert rows[0]["error"] is None
    assert (
        complete_module.complete_prompts("a", **kwargs)[0]["generated_text"]
        == rows[0]["generated_text"]
    )
    assert len(provider.created) == 1


def test_force_refresh_recovers_persisted_error_and_next_call_uses_success(refresh_client):
    provider, client, kwargs = refresh_client
    provider.fail_indices = {0}
    failed = complete_module.complete_prompts("a", **kwargs)
    assert len(provider.created) == 1
    assert failed[0]["error"] == "request failed"
    assert failed[0]["generated_text"] is None
    # Errors persist with the original batch; only successes enter response_cache.
    failed_job_id = failed[0]["batch_job_id"]
    persisted = Store(client.storage_path)
    assert persisted.results(failed_job_id)[0].error == "request failed"
    key = request_cache_hash(
        provider.name,
        effective_request_body(
            provider.name, {"model": "m", "messages": [{"role": "user", "content": "a"}]}
        ),
    )
    assert persisted.cached_responses(key) == []

    provider.fail_indices.clear()
    refreshed = complete_module.complete_prompts("a", force_refresh=True, **kwargs)
    assert len(provider.created) == 2
    assert refreshed[0]["batch_job_id"] != failed_job_id
    assert refreshed[0]["error"] is None
    assert refreshed[0]["generated_text"] is not None
    stored = persisted.cached_responses(key)
    assert len(stored) == 1
    assert stored[0].error is None
    assert stored[0].response == refreshed[0]["response"]

    # A fresh client must read the successful response from SQLite without
    # submitting or reusing the original batch that still contains the error.
    restarted = BatchClient(provider, storage_path=client.storage_path)

    def reject_submission(*args, **kwargs):
        raise AssertionError("third call must retrieve the refreshed response from cache")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(complete_module, "BatchClient", lambda *args, **kwargs: restarted)
        patch.setattr(restarted, "submit", reject_submission)
        cached = complete_module.complete_prompts("a", **kwargs)

    assert len(cached) == 1
    assert len(provider.created) == 2
    assert cached[0]["batch_job_id"] is None
    for field in [
        "response",
        "generated_text",
        "error",
        "error_message",
        "usage",
        "usage_scope",
        "usage_id",
    ]:
        assert cached[0][field] == refreshed[0][field]


def test_failed_refresh_preserves_cache_and_returns_errors(refresh_client):
    provider, _, kwargs = refresh_client
    original = complete_module.complete_prompts("a", **kwargs)
    provider.fail_indices = {0}
    failed = complete_module.complete_prompts("a", force_refresh=True, **kwargs)
    assert failed[0]["error"] == "request failed"
    assert failed[0]["generated_text"] is None
    assert (
        complete_module.complete_prompts("a", **kwargs)[0]["generated_text"]
        == original[0]["generated_text"]
    )
    assert len(provider.created) == 2


def test_partial_refresh_caches_only_successful_new_samples(refresh_client):
    provider, client, kwargs = refresh_client
    complete_module.complete_prompts("a", min_repeat=3, **kwargs)
    provider.fail_indices = {1}
    refreshed = complete_module.complete_prompts("a", min_repeat=3, force_refresh=True, **kwargs)
    assert [row["error"] for row in refreshed] == [None, "request failed", None]
    cached = complete_module.complete_prompts("a", min_repeat=2, **kwargs)
    assert [row["generated_text"] for row in cached] == [
        refreshed[i]["generated_text"] for i in [0, 2]
    ]
    key = request_cache_hash(
        provider.name,
        effective_request_body(
            provider.name, {"model": "m", "messages": [{"role": "user", "content": "a"}]}
        ),
    )
    assert len(client.store.cached_responses(key)) == 2
    assert len(provider.created) == 2


def test_failed_batch_refresh_preserves_cache(refresh_client):
    provider, _, kwargs = refresh_client
    original = complete_module.complete_prompts("a", **kwargs)
    provider.batch_status = BatchStatus.FAILED
    with pytest.raises(RuntimeError, match="ended with status=failed"):
        complete_module.complete_prompts("a", force_refresh=True, **kwargs)
    assert (
        complete_module.complete_prompts("a", **kwargs)[0]["generated_text"]
        == original[0]["generated_text"]
    )


def test_refresh_jobs_remain_resumable_by_id(refresh_client):
    provider, client, _ = refresh_client
    original = client.submit("a", model="m")
    refreshed = client.submit("a", model="m", force_refresh=True)
    assert original.id != refreshed.id
    # Resume the explicitly selected refresh job after a new client is constructed.
    restarted = BatchClient(provider, storage_path=client.storage_path)
    assert restarted.wait(refreshed.id).completed
    assert len(provider.created) == 2
    assert restarted.submit("a", model="m").id == original.id


def test_replace_cached_responses_preserves_metadata_and_ignores_errors(tmp_path: Path):
    store = Store(tmp_path)
    store.add_cached_response("key", {"old": True})
    store.add_cached_response("other", {"keep": True})
    results = [
        BatchResult("r0", {"new": 1}, usage={"total_tokens": 2}, usage_id="job:r0"),
        BatchResult("r1", {"invalid": True}, error="failed"),
        BatchResult("r2", None),
        BatchResult("r3", {"new": 2}),
    ]
    store.replace_cached_responses("key", results)
    rows = store.cached_responses("key")
    assert [row.response for row in rows] == [{"new": 1}, {"new": 2}]
    assert [row.custom_id for row in rows] == ["cache-0", "cache-1"]
    assert rows[0].usage == {"total_tokens": 2}
    assert rows[0].usage_scope == "request"
    assert rows[0].usage_id == "job:r0"
    assert store.cached_responses("other")[0].response == {"keep": True}
    store.replace_cached_responses("key", results[1:3])
    assert store.cached_responses("key") == rows


def test_replace_cached_responses_rolls_back_on_insert_failure(tmp_path: Path):
    import sqlite3

    store = Store(tmp_path)
    store.add_cached_response("key", {"old": True})
    with store.immediate() as conn:
        conn.execute(
            "CREATE TRIGGER reject_cache_insert BEFORE INSERT ON response_cache "
            "BEGIN SELECT RAISE(ABORT, 'insert rejected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="insert rejected"):
        store.replace_cached_responses("key", [BatchResult("r0", {"new": True})])
    assert store.cached_responses("key")[0].response == {"old": True}
