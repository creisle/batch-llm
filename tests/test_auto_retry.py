from types import SimpleNamespace

import pytest

import batch_llm.client as client_module
from batch_llm.models import BatchStatus


def test_reconcile_submission_retries_before_giving_up(monkeypatch):
    job = SimpleNamespace(
        id="job-1",
        remote_job_id=None,
        remote_file_id="file-1",
        status=BatchStatus.SUBMITTING,
        error=None,
    )

    class Store:
        def get_job(self, job_id):
            assert job_id == job.id
            return job

        def update_job(self, job_id, **fields):
            assert job_id == job.id
            for key, value in fields.items():
                setattr(job, key, value)

    attempts = iter([None, None, "batch-1"])
    provider = SimpleNamespace(
        name="openai",
        find_created_job=lambda *, job_id: next(attempts),
        find_created_job_by_file_id=lambda *, file_id: None,
    )
    sleeps = []
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(client_module, "_RECONCILE_DELAYS", (0, 1, 2))

    client = object.__new__(client_module.BatchClient)
    client.provider = provider
    client.store = Store()

    recovered = client._reconcile_submission(job, retry=True)

    assert recovered.remote_job_id == "batch-1"
    assert recovered.status == BatchStatus.SUBMITTED
    assert sleeps == [1, 2]


def test_refresh_automatically_retries_uncertain_submission_once(monkeypatch):
    job = SimpleNamespace(
        id="job-1",
        remote_job_id=None,
        remote_file_id="file-existing",
        status=BatchStatus.SUBMITTING,
        error=None,
    )

    class Store:
        def get_job(self, job_id):
            assert job_id == job.id
            return job

        def has_active_lease(self, job_id):
            return False

        def update_job(self, job_id, **fields):
            assert job_id == job.id
            for key, value in fields.items():
                setattr(job, key, value)

    client = object.__new__(client_module.BatchClient)
    client.provider = SimpleNamespace()
    client.store = Store()

    monkeypatch.setattr(client, "_reconcile_submission", lambda current, retry=False: current)

    def ensure_submitted(job_id):
        assert job_id == job.id
        assert job.status == BatchStatus.PREPARING
        assert job.error == client_module._AUTO_RETRY_MARKER
        job.remote_job_id = "batch-new"
        job.status = BatchStatus.SUBMITTED
        job.error = None
        return job

    monkeypatch.setattr(client, "_ensure_submitted", ensure_submitted)
    client.provider.status = lambda remote_job_id: (BatchStatus.SUBMITTED, None)

    recovered = client.refresh(job.id)

    assert recovered.remote_job_id == "batch-new"
    assert recovered.error is None


def test_refresh_does_not_automatically_retry_twice(monkeypatch):
    job = SimpleNamespace(
        id="job-1",
        remote_job_id=None,
        remote_file_id="file-existing",
        status=BatchStatus.SUBMITTING,
        error=client_module._AUTO_RETRY_MARKER,
    )

    class Store:
        def get_job(self, job_id):
            return job

        def has_active_lease(self, job_id):
            return False

    client = object.__new__(client_module.BatchClient)
    client.provider = SimpleNamespace()
    client.store = Store()

    monkeypatch.setattr(client, "_reconcile_submission", lambda current, retry=False: current)
    monkeypatch.setattr(
        client,
        "_ensure_submitted",
        lambda job_id: (_ for _ in ()).throw(AssertionError("must not retry twice")),
    )

    with pytest.raises(client_module.SubmissionUncertainError):
        client.refresh(job.id)
