from pathlib import Path

import pytest

from batch_llm.client import BatchClient, SubmissionUncertainError
from batch_llm.models import BatchResult, BatchStatus
from batch_llm.providers.base import Provider


class FakeProvider(Provider):
    name = "fake"

    def __init__(self):
        self.created = 0
        self.remote = {}

    def write_input(self, requests, path):
        path.write_text("input")

    def upload(self, path):
        return "file-1"

    def create(self, *, job_id, model, file_id):
        self.created += 1
        remote_id = f"remote-{job_id}"
        self.remote[job_id] = remote_id
        return remote_id

    def find_created_job(self, *, job_id):
        return self.remote.get(job_id)

    def status(self, remote_job_id):
        return BatchStatus.COMPLETED, None

    def results(self, remote_job_id):
        return [BatchResult("request-0", {"ok": True})]


def test_same_submission_does_not_resubmit(tmp_path: Path):
    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    first = client.submit("hello", model="m")
    second = client.submit("hello", model="m")
    assert first.id == second.id
    assert provider.created == 1


def test_recovers_remote_job_after_crash_window(tmp_path: Path):
    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")

    # Simulate local loss of the remote id after provider create succeeded.
    with client.store.immediate() as conn:
        conn.execute(
            "UPDATE jobs SET remote_job_id=NULL, status=? WHERE id=?",
            (BatchStatus.SUBMITTING.value, job.id),
        )

    recovered = client.refresh(job.id)
    assert recovered.remote_job_id == f"remote-{job.id}"
    assert provider.created == 1


def test_uncertain_job_is_automatically_resubmitted_once(tmp_path: Path, monkeypatch):
    import batch_llm.client as client_module

    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")

    provider.remote.clear()
    with client.store.immediate() as conn:
        conn.execute(
            "UPDATE jobs SET remote_job_id=NULL, status=? WHERE id=?",
            (BatchStatus.SUBMITTING.value, job.id),
        )

    monkeypatch.setattr(client_module, "_RECONCILE_DELAYS", (0,))
    recovered = client.refresh(job.id)
    assert recovered.remote_job_id == f"remote-{job.id}"
    assert provider.created == 2


def test_heartbeat_prevents_takeover_during_slow_submission(tmp_path: Path):
    import threading
    import time

    class SlowProvider(FakeProvider):
        def create(self, *, job_id, model, file_id):
            self.created += 1
            time.sleep(2.0)
            remote_id = f"remote-{job_id}"
            self.remote[job_id] = remote_id
            return remote_id

    provider = SlowProvider()
    first = BatchClient(provider, storage_path=tmp_path, lease_seconds=1)
    second = BatchClient(provider, storage_path=tmp_path, lease_seconds=1)
    errors = []

    def submit_first():
        try:
            first.submit("hello", model="m")
        except Exception as exc:  # pragma: no cover - diagnostic guard
            errors.append(exc)

    thread = threading.Thread(target=submit_first)
    thread.start()
    time.sleep(1.2)  # past the original lease; heartbeat should have renewed it
    second.submit("hello", model="m")
    thread.join()

    assert not errors
    assert provider.created == 1


def test_invalid_provider_name_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="provider must"):
        BatchClient("other", storage_path=tmp_path)


def test_storage_path_property_and_empty_submit(tmp_path: Path):
    client = BatchClient(FakeProvider(), storage_path=tmp_path)
    assert client.storage_path == tmp_path / "state.sqlite3"
    with pytest.raises(ValueError, match="cannot be empty"):
        client.submit([], model="m")


def test_submit_includes_system_prompt_and_generation(tmp_path: Path):
    class CaptureProvider(FakeProvider):
        def write_input(self, requests, path):
            self.requests = requests
            super().write_input(requests, path)

    provider = CaptureProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    client.submit("hello", model="m", generation={"temperature": 0.2}, system_prompt="system")
    body = provider.requests[0][1]
    assert body == {
        "model": "m",
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "hello"}],
        "temperature": 0.2,
    }


def test_refresh_preparing_job_submits_it(tmp_path: Path):
    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.store.create_or_get_job(
        job_id="manual",
        provider="fake",
        model="m",
        endpoint="e",
        request_hash="manual-hash",
        requests=[("request-0", {"model": "m", "messages": []})],
    )
    refreshed = client.refresh(job)
    assert refreshed.remote_job_id == "remote-manual"
    assert provider.created == 1


def test_wait_returns_terminal_job_immediately(tmp_path: Path):
    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")
    client.store.update_job(job.id, status=BatchStatus.COMPLETED)
    waited = client.wait(job.id, poll_interval=0)
    assert waited.status == BatchStatus.COMPLETED


def test_results_return_empty_for_noncompleted_job(tmp_path: Path):
    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")
    assert client.results(job, refresh=False) == []


def test_results_use_persisted_rows_without_downloading_again(tmp_path: Path):
    class CountingProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.result_calls = 0

        def results(self, remote_job_id):
            self.result_calls += 1
            return super().results(remote_job_id)

    provider = CountingProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")
    completed = client.refresh(job)
    assert completed.status == BatchStatus.COMPLETED
    assert provider.result_calls == 1
    assert client.results(job, refresh=False)
    assert provider.result_calls == 1


def test_get_returns_job(tmp_path: Path):
    client = BatchClient(FakeProvider(), storage_path=tmp_path)
    job = client.submit("hello", model="m")
    assert client.get(job.id).id == job.id


def test_retry_uncertain_without_force_raises(tmp_path: Path):
    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")
    provider.remote.clear()
    with client.store.immediate() as conn:
        conn.execute(
            "UPDATE jobs SET remote_job_id=NULL, status=?, lease_owner=NULL, lease_expires=NULL WHERE id=?",
            (BatchStatus.SUBMITTING.value, job.id),
        )
    with pytest.raises(SubmissionUncertainError):
        client.retry_uncertain_submission(job.id)


def test_retry_uncertain_returns_reconciled_remote_job(tmp_path: Path):
    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.submit("hello", model="m")
    with client.store.immediate() as conn:
        conn.execute(
            "UPDATE jobs SET remote_job_id=NULL, status=?, lease_owner=NULL, lease_expires=NULL WHERE id=?",
            (BatchStatus.SUBMITTING.value, job.id),
        )
    recovered = client.retry_uncertain_submission(job.id)
    assert recovered.remote_job_id == f"remote-{job.id}"
    assert provider.created == 1


def test_retry_non_uncertain_job_is_noop(tmp_path: Path):
    provider = FakeProvider()
    client = BatchClient(provider, storage_path=tmp_path)
    job = client.store.create_or_get_job(
        job_id="manual",
        provider="fake",
        model="m",
        endpoint="e",
        request_hash="hash-manual",
        requests=[("request-0", {})],
    )
    result = client.retry_uncertain_submission(job)
    assert result.id == job.id
    assert provider.created == 0


def test_batch_job_identity_ignores_provider_ignored_generation_options(tmp_path):
    class IdentityProvider(FakeProvider):
        def cache_identity_body(self, body):
            return {k: v for k, v in body.items() if k != "min_new_tokens"}

    provider = IdentityProvider()
    client = BatchClient(provider, storage_path=tmp_path / "state.sqlite3")
    job_a = client.submit(["hello"], model="model", generation={"temperature": 0})
    job_b = client.submit(
        ["hello"], model="model", generation={"temperature": 0, "min_new_tokens": 5}
    )

    assert job_a.id == job_b.id
    assert provider.created == 1
