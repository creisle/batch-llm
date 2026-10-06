from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .models import BatchJob, BatchResult, BatchStatus
from .providers.base import Provider
from .storage import Store


class SubmissionUncertainError(RuntimeError):
    pass


class BatchClient:
    """Crash-safe batch client for OpenAI and Gemini.

    Re-submission is conservative by design. If a process dies after the remote
    create call may have succeeded but before the returned job id was persisted,
    the client first searches the provider for the deterministic local marker.
    If it cannot prove whether a job exists, it leaves the job in SUBMITTING and
    raises SubmissionUncertainError rather than creating a duplicate batch.
    """

    def __init__(
        self,
        provider: str | Provider,
        *,
        api_key: str | None = None,
        storage_path: str | os.PathLike[str] | None = None,
        lease_seconds: int = 300,
    ):
        self.store = Store(storage_path)
        self.lease_seconds = lease_seconds
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

        if isinstance(provider, str):
            name = provider.lower()
            if name == "openai":
                from .providers.openai import OpenAIProvider
                self.provider = OpenAIProvider(api_key=api_key)
            elif name == "gemini":
                from .providers.gemini import GeminiProvider
                self.provider = GeminiProvider(api_key=api_key)
            else:
                raise ValueError("provider must be 'openai', 'gemini', or a Provider instance")
        else:
            self.provider = provider

    @property
    def storage_path(self) -> Path:
        return self.store.path

    def submit(
        self,
        prompts: Iterable[str] | str,
        *,
        model: str,
        generation: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> BatchJob:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if not prompts:
            raise ValueError("prompts cannot be empty")

        generation = dict(generation or {})
        requests: list[tuple[str, dict[str, Any]]] = []
        for i, prompt in enumerate(prompts):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            body = {"model": model, "messages": messages, **generation}
            requests.append((f"request-{i}", body))

        identity_requests = [
            (custom_id, self.provider.cache_identity_body(body))
            for custom_id, body in requests
        ]
        request_hash = self._hash_request(self.provider.name, model, identity_requests)
        job_id = request_hash[:32]
        job = self.store.create_or_get_job(
            job_id=job_id,
            provider=self.provider.name,
            model=model,
            endpoint="/v1/chat/completions",
            request_hash=request_hash,
            requests=requests,
        )

        if job.status.terminal or job.remote_job_id:
            return job
        return self._ensure_submitted(job.id)

    def refresh(self, job_or_id: BatchJob | str) -> BatchJob:
        job = self.store.get_job(job_or_id.id if isinstance(job_or_id, BatchJob) else job_or_id)

        if job.status == BatchStatus.PREPARING and not job.remote_job_id:
            job = self._ensure_submitted(job.id)

        if job.status == BatchStatus.SUBMITTING and not job.remote_job_id:
            # Another live worker may currently be inside the remote create call.
            # Do not mistake an in-flight submission for a crashed one.
            if self.store.has_active_lease(job.id):
                return job
            job = self._reconcile_submission(job)
            if not job.remote_job_id:
                raise SubmissionUncertainError(
                    f"job {job.id} may have been submitted remotely but its remote id was not persisted; "
                    "no matching remote job was found, so it will not be resubmitted automatically"
                )

        if not job.remote_job_id or job.status.terminal:
            return job

        status, error = self.provider.status(job.remote_job_id)
        self.store.update_job(job.id, status=status, error=error)

        if status == BatchStatus.COMPLETED:
            results = self.provider.results(job.remote_job_id)
            self.store.save_results(job.id, results)

        return self.store.get_job(job.id)

    def wait(
        self,
        job_or_id: BatchJob | str,
        *,
        poll_interval: float = 30,
        timeout: float | None = None,
    ) -> BatchJob:
        job_id = job_or_id.id if isinstance(job_or_id, BatchJob) else job_or_id
        started = time.monotonic()
        while True:
            job = self.refresh(job_id)
            if job.status.terminal:
                return job
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f"timed out waiting for batch {job_id}")
            time.sleep(poll_interval)

    def results(self, job_or_id: BatchJob | str, *, refresh: bool = True) -> list[BatchResult]:
        job_id = job_or_id.id if isinstance(job_or_id, BatchJob) else job_or_id
        job = self.refresh(job_id) if refresh else self.store.get_job(job_id)
        if job.status != BatchStatus.COMPLETED:
            return []
        cached = self.store.results(job_id)
        if cached:
            return cached
        if not job.remote_job_id:
            return []
        results = self.provider.results(job.remote_job_id)
        self.store.save_results(job_id, results)
        return results

    def get(self, job_id: str) -> BatchJob:
        return self.store.get_job(job_id)

    def retry_uncertain_submission(self, job_or_id: BatchJob | str, *, force: bool = False) -> BatchJob:
        """Retry a job stuck in SUBMITTING.

        First reconciles against remote jobs. A new batch is created only when
        force=True and no matching remote job can be found.
        """
        job_id = job_or_id.id if isinstance(job_or_id, BatchJob) else job_or_id
        job = self._reconcile_submission(self.store.get_job(job_id))
        if job.remote_job_id:
            return job
        if job.status != BatchStatus.SUBMITTING:
            return job
        if not force:
            raise SubmissionUncertainError(
                f"job {job.id} is still submission-uncertain; pass force=True only after confirming "
                "that no remote batch exists"
            )
        self.store.update_job(job.id, status=BatchStatus.PREPARING)
        return self._ensure_submitted(job.id)

    def _ensure_submitted(self, job_id: str) -> BatchJob:
        if not self.store.claim(job_id, self.owner, self.lease_seconds):
            return self.store.get_job(job_id)
        try:
            with self._lease_heartbeat(job_id):
                return self._submit_while_leased(job_id)
        finally:
            self.store.release(job_id, self.owner)

    def _submit_while_leased(self, job_id: str) -> BatchJob:
        job = self.store.get_job(job_id)
        if job.remote_job_id or job.status.terminal:
            return job
        if job.status == BatchStatus.SUBMITTING:
            return self._reconcile_submission(job)

        requests = self.store.request_rows(job.id)
        work_dir = self.store.path.parent / "inputs"
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / f"{job.id}.jsonl"

        self.provider.write_input(requests, input_path)

        # Persist uploads so a restarted process can reuse large input files.
        remote_file_id = job.remote_file_id
        if not remote_file_id:
            remote_file_id = self.provider.upload(input_path)
            self.store.update_job(job.id, remote_file_id=remote_file_id)

        # A crash after this state transition may have created the remote job.
        # Recovery must reconcile first and must never blindly create again.
        self.store.update_job(job.id, status=BatchStatus.SUBMITTING)
        remote_job_id = self.provider.create(
            job_id=job.id,
            model=job.model,
            file_id=remote_file_id,
        )
        self.store.update_job(
            job.id,
            remote_job_id=remote_job_id,
            status=BatchStatus.SUBMITTED,
        )
        return self.store.get_job(job.id)

    @contextmanager
    def _lease_heartbeat(self, job_id: str):
        stop = threading.Event()
        interval = max(1.0, self.lease_seconds / 3)

        def heartbeat() -> None:
            while not stop.wait(interval):
                if not self.store.renew(job_id, self.owner, self.lease_seconds):
                    return

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=interval + 1)

    def _reconcile_submission(self, job: BatchJob) -> BatchJob:
        if job.remote_job_id:
            return job
        if job.status != BatchStatus.SUBMITTING:
            return job
        remote_job_id = self.provider.find_created_job(job_id=job.id)
        if remote_job_id:
            self.store.update_job(
                job.id,
                remote_job_id=remote_job_id,
                status=BatchStatus.SUBMITTED,
            )
        return self.store.get_job(job.id)

    @staticmethod
    def _hash_request(
        provider: str,
        model: str,
        requests: list[tuple[str, dict[str, Any]]],
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "requests": requests,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
