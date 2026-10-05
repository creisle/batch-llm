import multiprocessing
import sqlite3
from pathlib import Path

from batch_llm.storage import Store


def _claim_worker(db_path: str, start, queue, owner: str) -> None:
    store = Store(db_path)
    start.wait()
    queue.put(store.claim("job", owner, lease_seconds=60))


def _cache_worker(db_path: str, start, queue, value: int) -> None:
    store = Store(db_path)
    start.wait()
    index = store.add_cached_response("key", {"value": value})
    queue.put(index)


def test_only_one_process_can_claim_same_job(tmp_path: Path):
    ctx = multiprocessing.get_context("spawn")
    db = tmp_path / "state.sqlite3"
    store = Store(db)
    store.create_or_get_job(
        job_id="job",
        provider="fake",
        model="m",
        endpoint="e",
        request_hash="hash",
        requests=[("request-0", {})],
    )

    start = ctx.Event()
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_claim_worker, args=(str(db), start, queue, f"worker-{i}"))
        for i in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7


def test_concurrent_cache_writes_get_unique_contiguous_sample_indices(tmp_path: Path):
    ctx = multiprocessing.get_context("spawn")
    db = tmp_path / "state.sqlite3"
    Store(db)
    start = ctx.Event()
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=_cache_worker, args=(str(db), start, queue, i))
        for i in range(12)
    ]
    for process in processes:
        process.start()
    start.set()
    indices = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert sorted(indices) == list(range(12))
    rows = Store(db).cached_responses("key")
    assert len(rows) == 12
    assert sorted(row.response["value"] for row in rows) == list(range(12))


def test_expired_lease_can_be_reclaimed(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    store.create_or_get_job(
        job_id="job",
        provider="fake",
        model="m",
        endpoint="e",
        request_hash="hash",
        requests=[("request-0", {})],
    )
    assert store.claim("job", "dead-worker", 60)
    with store.immediate() as conn:
        conn.execute("UPDATE jobs SET lease_expires = 0 WHERE id = 'job'")
    assert store.claim("job", "new-worker", 60)


def test_store_enables_wal_and_busy_timeout(tmp_path: Path):
    store = Store(tmp_path / "state.sqlite3")
    conn = store.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_existing_database_schema_is_upgraded_in_place(tmp_path: Path):
    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            request_hash TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            remote_file_id TEXT,
            remote_job_id TEXT,
            error TEXT,
            lease_owner TEXT,
            lease_expires REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE requests (
            job_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            custom_id TEXT NOT NULL,
            body TEXT NOT NULL,
            PRIMARY KEY(job_id, ordinal),
            UNIQUE(job_id, custom_id)
        );
        CREATE TABLE results (
            job_id TEXT NOT NULL,
            custom_id TEXT NOT NULL,
            response TEXT,
            error TEXT,
            PRIMARY KEY(job_id, custom_id)
        );
        CREATE TABLE response_cache (
            request_hash TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(request_hash, sample_index)
        );
        """
    )
    conn.close()

    store = Store(db)
    conn = store.connect()
    try:
        result_columns = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
        cache_columns = {row[1] for row in conn.execute("PRAGMA table_info(response_cache)")}
    finally:
        conn.close()
    assert {"usage", "usage_scope"} <= result_columns
    assert {"usage", "usage_scope", "usage_id"} <= cache_columns
