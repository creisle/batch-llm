from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..models import BatchResult, BatchStatus
from ..request_identity import openai_request_body
from .base import Provider


class OpenAIProvider(Provider):
    name = "openai"

    @staticmethod
    def _request_body(body: dict[str, Any]) -> dict[str, Any]:
        return openai_request_body(body)

    def cache_identity_body(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request_body(body)

    def __init__(self, api_key: str | None = None, client: OpenAI | None = None):
        self.client = client or OpenAI(api_key=api_key)

    def write_input(self, requests: list[tuple[str, dict[str, Any]]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for custom_id, body in requests:
                json.dump(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": self._request_body(body),
                    },
                    fh,
                    separators=(",", ":"),
                )
                fh.write("\n")

    def upload(self, path: Path) -> str:
        with path.open("rb") as fh:
            return self.client.files.create(file=fh, purpose="batch").id

    def create(self, *, job_id: str, model: str, file_id: str) -> str:
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"batch_llm_job": job_id},
        )
        return batch.id

    def find_created_job(self, *, job_id: str) -> str | None:
        # The OpenAI SDK's page iterator follows pagination, so this is not limited
        # to the first page of recent jobs.
        for batch in self.client.batches.list(limit=100):
            metadata = getattr(batch, "metadata", None) or {}
            if metadata.get("batch_llm_job") == job_id:
                return batch.id
        return None

    def status(self, remote_job_id: str) -> tuple[BatchStatus, str | None]:
        batch = self.client.batches.retrieve(remote_job_id)
        raw = batch.status
        mapping = {
            "validating": BatchStatus.SUBMITTED,
            "in_progress": BatchStatus.RUNNING,
            "finalizing": BatchStatus.RUNNING,
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "expired": BatchStatus.EXPIRED,
            "cancelling": BatchStatus.RUNNING,
            "cancelled": BatchStatus.CANCELLED,
        }
        error = None
        errors = getattr(batch, "errors", None)
        if errors:
            error = str(errors)
        return mapping.get(raw, BatchStatus.UNKNOWN), error

    def results(self, remote_job_id: str) -> list[BatchResult]:
        batch = self.client.batches.retrieve(remote_job_id)
        rows: list[BatchResult] = []

        # Successful and failed requests may be split across output and error
        # files. Read both so a completed batch always maps back to every
        # request that the provider reported.
        file_ids = [getattr(batch, "output_file_id", None), getattr(batch, "error_file_id", None)]
        for file_id in dict.fromkeys(file_id for file_id in file_ids if file_id):
            content = self.client.files.content(file_id).content
            for line in content.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                response = item.get("response")
                body = response.get("body") if response else None
                usage = (body or {}).get("usage")
                error = item.get("error")
                if error is None and isinstance(body, dict):
                    error = body.get("error")
                rows.append(
                    BatchResult(
                        custom_id=item["custom_id"],
                        response=body,
                        error=error,
                        usage=usage,
                        usage_scope="request" if usage else None,
                    )
                )
        return rows
