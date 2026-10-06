from pathlib import Path

from batch_llm.models import BatchStatus
from batch_llm.storage import Store


def test_create_same_request_is_idempotent(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    reqs = [("request-0", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})]
    a = store.create_or_get_job(
        job_id="a",
        provider="fake",
        model="x",
        endpoint="/v1/chat/completions",
        request_hash="hash",
        requests=reqs,
    )
    b = store.create_or_get_job(
        job_id="b",
        provider="fake",
        model="x",
        endpoint="/v1/chat/completions",
        request_hash="hash",
        requests=reqs,
    )
    assert a.id == b.id == "a"


def test_claim_is_exclusive(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    store.create_or_get_job(
        job_id="a", provider="fake", model="x", endpoint="e",
        request_hash="hash", requests=[("r", {})]
    )
    assert store.claim("a", "one", 60)
    assert not store.claim("a", "two", 60)
    store.release("a", "one")
    assert store.claim("a", "two", 60)


def test_results_follow_request_order_not_provider_order(tmp_path: Path):
    from batch_llm.models import BatchResult

    store = Store(tmp_path / "state.sqlite3")
    store.create_or_get_job(
        job_id="a",
        provider="fake",
        model="x",
        endpoint="e",
        request_hash="hash",
        requests=[("request-0", {}), ("request-1", {})],
    )
    store.save_results(
        "a",
        [
            BatchResult("request-1", {"value": 1}),
            BatchResult("request-0", {"value": 0}),
        ],
    )
    assert [r.custom_id for r in store.results("a")] == ["request-0", "request-1"]


def test_results_preserve_usage(tmp_path: Path):
    from batch_llm.models import BatchResult

    store = Store(tmp_path / "state.sqlite3")
    store.create_or_get_job(
        job_id="usage",
        provider="fake",
        model="x",
        endpoint="e",
        request_hash="usage-hash",
        requests=[("request-0", {})],
    )
    usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    store.save_results(
        "usage",
        [BatchResult("request-0", {"usage": usage}, usage=usage, usage_scope="request")],
    )
    result = store.results("usage")[0]
    assert result.usage == usage
    assert result.response["usage"] == usage
    assert result.usage_scope == "request"
    assert result.usage_id == "usage:request-0"


def test_response_cache_preserves_sample_order_and_usage(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    store.add_cached_response(
        "k",
        {"value": 1},
        {"total_tokens": 10},
        usage_scope="request",
        usage_id="job:request-0",
    )
    store.add_cached_response("k", {"value": 2}, None)
    rows = store.cached_responses("k")
    assert [r.response["value"] for r in rows] == [1, 2]
    assert rows[0].usage == {"total_tokens": 10}
    assert rows[0].usage_scope == "request"
    assert rows[0].usage_id == "job:request-0"
    assert rows[1].usage is None
    assert rows[1].usage_id is None


def test_get_missing_job_raises(tmp_path: Path):
    import pytest

    store = Store(tmp_path / "state.sqlite3")
    with pytest.raises(KeyError):
        store.get_job("missing")


def test_update_job_rejects_invalid_fields_and_empty_update(tmp_path: Path):
    import pytest

    store = Store(tmp_path / "state.sqlite3")
    store.create_or_get_job(
        job_id="a", provider="fake", model="x", endpoint="e", request_hash="hash-x", requests=[]
    )
    store.update_job("a")
    with pytest.raises(ValueError, match="invalid job fields"):
        store.update_job("a", nope=1)


def test_decode_plain_text_error(tmp_path: Path):
    from batch_llm.models import BatchResult

    store = Store(tmp_path / "state.sqlite3")
    store.create_or_get_job(
        job_id="err",
        provider="fake",
        model="x",
        endpoint="e",
        request_hash="hash-err",
        requests=[("request-0", {})],
    )
    store.save_results("err", [BatchResult("request-0", None, error="plain error")])
    assert store.results("err")[0].error == "plain error"


def test_cache_request_provenance_is_persisted(tmp_path: Path):
    import json
    import sqlite3

    store = Store(tmp_path / "state.sqlite3")
    body = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    store.ensure_cache_request("hash-1", "openai", body)
    store.add_cached_response("hash-1", {"choices": []})

    conn = sqlite3.connect(store.path)
    try:
        row = conn.execute(
            "SELECT provider, body FROM cache_requests WHERE request_hash = ?",
            ("hash-1",),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "openai"
    assert json.loads(row[1]) == body
