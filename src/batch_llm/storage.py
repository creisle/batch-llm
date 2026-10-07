from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator

from platformdirs import user_cache_dir

from .models import BatchJob, BatchResult, BatchStatus


def default_cache_dir() -> Path:
    override = os.environ.get("BATCH_LLM_CACHE_DIR")
    return Path(override).expanduser() if override else Path(user_cache_dir("batch-llm"))


class Store:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        if path is None:
            path = default_cache_dir() / "state.sqlite3"
        path = Path(path).expanduser()
        if path.suffix == "":
            path = path / "state.sqlite3"
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._setup()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _setup(self) -> None:
        with closing(self.connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
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

                CREATE UNIQUE INDEX IF NOT EXISTS jobs_remote_job
                    ON jobs(provider, remote_job_id)
                    WHERE remote_job_id IS NOT NULL;

                CREATE TABLE IF NOT EXISTS requests (
                    job_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    custom_id TEXT NOT NULL,
                    body TEXT NOT NULL,
                    PRIMARY KEY(job_id, ordinal),
                    UNIQUE(job_id, custom_id),
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS results (
                    job_id TEXT NOT NULL,
                    custom_id TEXT NOT NULL,
                    response TEXT,
                    error TEXT,
                    usage TEXT,
                    usage_scope TEXT,
                    PRIMARY KEY(job_id, custom_id),
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS cache_requests (
                    request_hash TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS response_cache (
                    request_hash TEXT NOT NULL,
                    sample_index INTEGER NOT NULL,
                    response TEXT NOT NULL,
                    usage TEXT,
                    usage_scope TEXT,
                    usage_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(request_hash, sample_index)
                );
                """
            )
            result_columns = {row["name"] for row in conn.execute("PRAGMA table_info(results)")}
            if "usage" not in result_columns:
                conn.execute("ALTER TABLE results ADD COLUMN usage TEXT")
            if "usage_scope" not in result_columns:
                conn.execute("ALTER TABLE results ADD COLUMN usage_scope TEXT")

            cache_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(response_cache)")
            }
            if "usage" not in cache_columns:
                conn.execute("ALTER TABLE response_cache ADD COLUMN usage TEXT")
            if "usage_scope" not in cache_columns:
                conn.execute("ALTER TABLE response_cache ADD COLUMN usage_scope TEXT")
            if "usage_id" not in cache_columns:
                conn.execute("ALTER TABLE response_cache ADD COLUMN usage_id TEXT")

    @contextmanager
    def immediate(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_or_get_job(
        self,
        *,
        job_id: str,
        provider: str,
        model: str,
        endpoint: str,
        request_hash: str,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> BatchJob:
        with self.immediate() as conn:
            conn.execute(
                """
                INSERT INTO jobs(id, provider, model, endpoint, request_hash, status)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_hash) DO NOTHING
                """,
                (job_id, provider, model, endpoint, request_hash, BatchStatus.PREPARING.value),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE request_hash = ?", (request_hash,)
            ).fetchone()
            assert row is not None
            existing_id = row["id"]
            if existing_id == job_id:
                conn.executemany(
                    """
                    INSERT INTO requests(job_id, ordinal, custom_id, body)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(job_id, ordinal) DO NOTHING
                    """,
                    [
                        (
                            job_id,
                            i,
                            custom_id,
                            json.dumps(body, sort_keys=True, separators=(",", ":")),
                        )
                        for i, (custom_id, body) in enumerate(requests)
                    ],
                )
            return self._row_to_job(row)

    def get_job(self, job_id: str) -> BatchJob:
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row_to_job(row)

    def claim(self, job_id: str, owner: str, lease_seconds: int = 300) -> bool:
        now = time.time()
        with self.immediate() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET lease_owner = ?, lease_expires = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND (lease_owner IS NULL OR lease_owner = ? OR lease_expires < ?)
                """,
                (owner, now + lease_seconds, job_id, owner, now),
            )
            return cur.rowcount == 1

    def renew(self, job_id: str, owner: str, lease_seconds: int = 300) -> bool:
        now = time.time()
        with self.immediate() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET lease_expires = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND lease_owner = ?
                """,
                (now + lease_seconds, job_id, owner),
            )
            return cur.rowcount == 1

    def has_active_lease(self, job_id: str) -> bool:
        now = time.time()
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT lease_owner, lease_expires FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return bool(
            row and row["lease_owner"] and row["lease_expires"] and row["lease_expires"] >= now
        )

    def release(self, job_id: str, owner: str) -> None:
        with self.immediate() as conn:
            conn.execute(
                """
                UPDATE jobs SET lease_owner = NULL, lease_expires = NULL,
                                updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND lease_owner = ?
                """,
                (job_id, owner),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "remote_file_id", "remote_job_id", "error"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"invalid job fields: {sorted(bad)}")
        if not fields:
            return
        if isinstance(fields.get("status"), BatchStatus):
            fields["status"] = fields["status"].value
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]
        with self.immediate() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                values,
            )

    def request_rows(self, job_id: str) -> list[tuple[str, dict[str, Any]]]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT custom_id, body FROM requests WHERE job_id = ? ORDER BY ordinal", (job_id,)
            ).fetchall()
        return [(r["custom_id"], json.loads(r["body"])) for r in rows]

    def save_results(self, job_id: str, results: list[BatchResult]) -> None:
        with self.immediate() as conn:
            conn.executemany(
                """
                INSERT INTO results(job_id, custom_id, response, error, usage, usage_scope)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, custom_id) DO UPDATE SET
                    response = excluded.response,
                    error = excluded.error,
                    usage = excluded.usage,
                    usage_scope = excluded.usage_scope
                """,
                [
                    (
                        job_id,
                        r.custom_id,
                        json.dumps(r.response, sort_keys=True) if r.response is not None else None,
                        json.dumps(r.error, sort_keys=True)
                        if isinstance(r.error, dict)
                        else r.error,
                        json.dumps(r.usage, sort_keys=True) if r.usage is not None else None,
                        r.usage_scope or ("request" if r.usage is not None else None),
                    )
                    for r in results
                ],
            )

    def ensure_cache_request(self, request_hash: str, provider: str, body: dict[str, Any]) -> None:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        with self.immediate() as conn:
            conn.execute(
                """
                INSERT INTO cache_requests(request_hash, provider, body)
                VALUES (?, ?, ?)
                ON CONFLICT(request_hash) DO UPDATE SET
                    provider = excluded.provider,
                    body = excluded.body
                """,
                (request_hash, provider.lower(), canonical),
            )

    def ensure_cache_requests(self, requests: list[tuple[str, str, dict[str, Any]]]) -> None:
        if not requests:
            return
        rows = [
            (
                request_hash,
                provider.lower(),
                json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            )
            for request_hash, provider, body in requests
        ]
        with self.immediate() as conn:
            conn.executemany(
                """
                INSERT INTO cache_requests(request_hash, provider, body)
                VALUES (?, ?, ?)
                ON CONFLICT(request_hash) DO NOTHING
                """,
                rows,
            )

    def cached_responses_many(self, request_hashes: list[str]) -> dict[str, list[BatchResult]]:
        unique_hashes = list(dict.fromkeys(request_hashes))
        if not unique_hashes:
            return {}
        rows_by_hash: dict[str, list[BatchResult]] = {key: [] for key in unique_hashes}
        with closing(self.connect()) as conn:
            # Stay below SQLite builds that still use the traditional 999-variable limit.
            for start in range(0, len(unique_hashes), 900):
                chunk = unique_hashes[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT request_hash, sample_index, response, usage, usage_scope, usage_id
                    FROM response_cache
                    WHERE request_hash IN ({placeholders})
                    ORDER BY request_hash, sample_index
                    """,
                    chunk,
                ).fetchall()
                for row in rows:
                    rows_by_hash[row["request_hash"]].append(
                        BatchResult(
                            custom_id=f"cache-{row['sample_index']}",
                            response=json.loads(row["response"]),
                            usage=json.loads(row["usage"]) if row["usage"] else None,
                            usage_scope=row["usage_scope"],
                            usage_id=row["usage_id"],
                        )
                    )
        return rows_by_hash

    def cached_responses(self, request_hash: str) -> list[BatchResult]:
        return self.cached_responses_many([request_hash])[request_hash]

    def add_cached_response(
        self,
        request_hash: str,
        response: dict[str, Any],
        usage: dict[str, Any] | None = None,
        *,
        usage_scope: str | None = None,
        usage_id: str | None = None,
    ) -> int:
        with self.immediate() as conn:
            next_index = conn.execute(
                "SELECT COALESCE(MAX(sample_index), -1) + 1 FROM response_cache WHERE request_hash = ?",
                (request_hash,),
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO response_cache(
                    request_hash, sample_index, response, usage, usage_scope, usage_id
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request_hash,
                    next_index,
                    json.dumps(response, sort_keys=True),
                    json.dumps(usage, sort_keys=True) if usage is not None else None,
                    usage_scope or ("request" if usage is not None else None),
                    usage_id,
                ),
            )
            return int(next_index)

    def results(self, job_id: str) -> list[BatchResult]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                SELECT r.custom_id, r.response, r.error, r.usage, r.usage_scope
                FROM results r
                LEFT JOIN requests q
                  ON q.job_id = r.job_id AND q.custom_id = r.custom_id
                WHERE r.job_id = ?
                ORDER BY q.ordinal, r.rowid
                """,
                (job_id,),
            ).fetchall()
        return [
            BatchResult(
                custom_id=r["custom_id"],
                response=json.loads(r["response"]) if r["response"] else None,
                error=self._decode_error(r["error"]),
                usage=json.loads(r["usage"]) if r["usage"] else None,
                usage_scope=r["usage_scope"],
                usage_id=(f"{job_id}:{r['custom_id']}" if r["usage"] else None),
            )
            for r in rows
        ]

    @staticmethod
    def _decode_error(value: str | None) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> BatchJob:
        return BatchJob(
            id=row["id"],
            provider=row["provider"],
            model=row["model"],
            status=BatchStatus(row["status"]),
            remote_file_id=row["remote_file_id"],
            remote_job_id=row["remote_job_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"],
        )
