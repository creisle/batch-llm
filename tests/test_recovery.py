import threading
import time
from pathlib import Path

import pytest

from batch_llm.client import BatchClient, SubmissionUncertainError
from batch_llm.models import BatchResult, BatchStatus
from batch_llm.providers.base import Provider


class RecordingProvider(Provider):
    name = "fake"

    def __init__(self):
        self.upload_count = 0
        self.create_count = 0
        self.result_count = 0
        self.remote = {}
        self.status_value = BatchStatus.COMPLETED

    def write_input(self, requests, path):
        path.write_text("input", encoding="utf-8")

    def upload(self, path):
        self.upload_count += 1
        return f"file-{self.upload_count}"

    def create(self, *, job_id, model, file_id):
        self.create_count += 1
        remote_id = f"remote-{job_id}"
        self.remote[job_id] = remote_id
        return remote_id

    def find_created_job(self, *, job_id):
        return self.remote.get(job_id)

    def status(self, remote_job_id):
        return self.status_value, None

    def results(self, remote_job_id):
        self.result_count += 1
        return [
            BatchResult(
                "request-0",
                {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 2}},
                usage={"total_tokens": 2},
                usage_scope="request",
            )
        ]


def test_uploaded_file_is_reused_after_failure_before_create(tmp_path: Path):
    class FailCreateOnce(RecordingProvider):
        def __init__(self):
            super().__init__()
            self.fail = True

        def create(self, *, job_id, model, file_id):
            if self.fail:
                self.fail = False
                raise ConnectionError("create failed before acceptance")
            return super().create(job_id=job_id, model=model, file_id=file_id)

    provider = FailCreateOnce()
    client = BatchClient(provider, storage_path=tmp_path)
    with pytest.raises(ConnectionError):
        client.submit("hello", model="m")

    conn = client.store.connect()
    try:
        jobs = conn.execute("SELECT id, remote_file_id, status FROM jobs").fetchall()
    finally:
        conn.close()
    assert len(jobs) == 1
    job_id = jobs[0]["id"]
    assert jobs[0]["remote_file_id"] == "file-1"
    assert jobs[0]["status"] == BatchStatus.SUBMITTING.value

    # The create call might have been accepted remotely, so normal recovery is conservative.
    with pytest.raises(SubmissionUncertainError):
        client.refresh(job_id)
    assert provider.upload_count == 1

    client.retry_uncertain_submission(job_id, force=True)
    assert provider.upload_count == 1
    assert provider.create_count == 1


def test_crash_after_remote_create_reconciles_without_second_create(tmp_path: Path, monkeypatch):
    provider = RecordingProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    original_update = client.store.update_job
    failed = False

    def crash_before_remote_id_persist(job_id, **fields):
        nonlocal failed
        if fields.get("remote_job_id") and not failed:
            failed = True
            raise RuntimeError("simulated process death")
        return original_update(job_id, **fields)

    monkeypatch.setattr(client.store, "update_job", crash_before_remote_id_persist)
    with pytest.raises(RuntimeError, match="simulated process death"):
        client.submit("hello", model="m")
    assert provider.create_count == 1

    # Fresh process/store view; provider can find the remotely accepted job.
    conn = client.store.connect()
    try:
        job_id = conn.execute("SELECT id FROM jobs").fetchone()[0]
    finally:
        conn.close()
    recovered = BatchClient(provider, storage_path=tmp_path).refresh(job_id)
    assert recovered.remote_job_id is not None
    assert provider.create_count == 1


def test_downloaded_results_are_recoverable_if_first_persist_fails(tmp_path: Path, monkeypatch):
    provider = RecordingProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")

    original_save = client.store.save_results
    calls = 0

    def fail_once(job_id, results):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("disk write interrupted")
        return original_save(job_id, results)

    monkeypatch.setattr(client.store, "save_results", fail_once)
    with pytest.raises(RuntimeError, match="disk write interrupted"):
        client.refresh(job.id)
    assert provider.result_count == 1

    rows = client.results(job.id)
    assert len(rows) == 1
    assert rows[0].response["choices"][0]["message"]["content"] == "ok"
    assert provider.result_count == 2


def test_active_lease_blocks_reconciliation_of_inflight_submission(tmp_path: Path):
    provider = RecordingProvider()
    first = BatchClient(provider, storage_path=tmp_path, lease_seconds=60)
    job = first.store.create_or_get_job(
        job_id="job",
        provider="fake",
        model="m",
        endpoint="e",
        request_hash="hash",
        requests=[("request-0", {"model": "m"})],
    )
    first.store.update_job(job.id, status=BatchStatus.SUBMITTING)
    assert first.store.claim(job.id, "other-worker", 60)

    second = BatchClient(provider, storage_path=tmp_path, lease_seconds=60)
    refreshed = second.refresh(job.id)
    assert refreshed.status == BatchStatus.SUBMITTING
    assert provider.create_count == 0


def test_wait_times_out_without_resubmitting(tmp_path: Path):
    provider = RecordingProvider()
    provider.status_value = BatchStatus.RUNNING
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")
    with pytest.raises(TimeoutError):
        client.wait(job, poll_interval=0.001, timeout=0.003)
    assert provider.create_count == 1
