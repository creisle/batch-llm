from pathlib import Path

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
        job_id="a",
        provider="fake",
        model="x",
        endpoint="e",
        request_hash="hash",
        requests=[("r", {})],
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
        "a", [BatchResult("request-1", {"value": 1}), BatchResult("request-0", {"value": 0})]
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
        "usage", [BatchResult("request-0", {"usage": usage}, usage=usage, usage_scope="request")]
    )
    result = store.results("usage")[0]
    assert result.usage == usage
    assert result.response["usage"] == usage
    assert result.usage_scope == "request"
    assert result.usage_id == "usage:request-0"


def test_response_cache_preserves_sample_order_and_usage(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    store.add_cached_response(
        "k", {"value": 1}, {"total_tokens": 10}, usage_scope="request", usage_id="job:request-0"
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
            "SELECT provider, body FROM cache_requests WHERE request_hash = ?", ("hash-1",)
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "openai"
    assert json.loads(row[1]) == body


def test_cached_responses_many_reads_multiple_keys(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    store.add_cached_response("a", {"value": 1})
    store.add_cached_response("a", {"value": 2})
    store.add_cached_response("b", {"value": 3})

    rows = store.cached_responses_many(["a", "b", "missing"])

    assert [row.response["value"] for row in rows["a"]] == [1, 2]
    assert [row.response["value"] for row in rows["b"]] == [3]
    assert rows["missing"] == []


def test_cached_responses_many_handles_more_than_sqlite_variable_limit(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    store.add_cached_response("key-1000", {"value": 1000})

    rows = store.cached_responses_many([f"key-{i}" for i in range(1001)])

    assert rows["key-0"] == []
    assert rows["key-1000"][0].response == {"value": 1000}


def test_ensure_cache_requests_inserts_in_one_bulk_operation(tmp_path: Path):
    import json
    import sqlite3

    store = Store(tmp_path / "state.sqlite3")
    store.ensure_cache_requests(
        [("hash-1", "openai", {"value": 1}), ("hash-2", "gemini", {"value": 2})]
    )
    with sqlite3.connect(store.path) as conn:
        rows = conn.execute(
            "SELECT request_hash, provider, body FROM cache_requests ORDER BY request_hash"
        ).fetchall()

    assert [(row[0], row[1], json.loads(row[2])) for row in rows] == [
        ("hash-1", "openai", {"value": 1}),
        ("hash-2", "gemini", {"value": 2}),
    ]


def test_bulk_cache_helpers(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    body1 = {"model": "m", "messages": [{"role": "user", "content": "one"}]}
    body2 = {"model": "m", "messages": [{"role": "user", "content": "two"}]}

    store.ensure_cache_requests([("k1", "openai", body1), ("k2", "openai", body2)])

    store.add_cached_response(
        "k1", {"value": 1}, {"total_tokens": 10}, usage_scope="request", usage_id="job:request-0"
    )
    store.add_cached_response("k1", {"value": 2})
    store.add_cached_response("k2", {"value": 3})

    # Include >900 keys so the SQLite chunking path is exercised.
    keys = ["k1", "k2", *[f"missing-{i}" for i in range(901)]]
    counts = store.cached_response_counts(keys)

    assert counts["k1"] == 2
    assert counts["k2"] == 1
    assert counts["missing-0"] == 0
    assert counts["missing-900"] == 0

    rows = store.cached_responses_many(["k1", "k2"], limit_per_key=1)
    assert [row.response["value"] for row in rows["k1"]] == [1]
    assert [row.response["value"] for row in rows["k2"]] == [3]
    assert rows["k1"][0].usage == {"total_tokens": 10}
    assert rows["k1"][0].usage_scope == "request"
    assert rows["k1"][0].usage_id == "job:request-0"

    # The legacy single-key method is still a wrapper around the bulk method.
    assert [row.response["value"] for row in store.cached_responses("k1")] == [1, 2]

    assert store.cached_response_counts([]) == {}
    assert store.cached_responses_many([]) == {}
    assert store.cached_responses_many(["k1"], limit_per_key=0) == {"k1": []}
