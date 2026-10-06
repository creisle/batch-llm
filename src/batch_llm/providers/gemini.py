from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from ..models import BatchResult, BatchStatus
from ..request_identity import gemini_request_body
from .base import Provider


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, client=None):
        self.client = client or genai.Client(api_key=api_key)

    @staticmethod
    def _request(body: dict[str, Any]) -> dict[str, Any]:
        return gemini_request_body(body)

    def cache_identity_body(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(body)

    def write_input(self, requests: list[tuple[str, dict[str, Any]]], path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for custom_id, body in requests:
                json.dump(
                    {"key": custom_id, "request": self._request(body)},
                    fh,
                    separators=(",", ":"),
                )
                fh.write("\n")

    def upload(self, path: Path) -> str:
        uploaded = self.client.files.upload(
            file=path,
            config=types.UploadFileConfig(mime_type="application/jsonl"),
        )
        return uploaded.name

    def create(self, *, job_id: str, model: str, file_id: str) -> str:
        batch = self.client.batches.create(
            model=model,
            src=file_id,
            config={"display_name": f"batch-llm-{job_id}"},
        )
        return batch.name

    def find_created_job(self, *, job_id: str) -> str | None:
        marker = f"batch-llm-{job_id}"
        for batch in self.client.batches.list():
            display_name = getattr(batch, "display_name", None)
            if display_name == marker:
                return batch.name
        return None

    def status(self, remote_job_id: str) -> tuple[BatchStatus, str | None]:
        batch = self.client.batches.get(name=remote_job_id)
        raw = str(batch.state).upper()
        mapping = {
            "JOB_STATE_PENDING": BatchStatus.SUBMITTED,
            "JOB_STATE_RUNNING": BatchStatus.RUNNING,
            "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
            "JOB_STATE_FAILED": BatchStatus.FAILED,
            "JOB_STATE_CANCELLED": BatchStatus.CANCELLED,
            "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
        }
        error = str(getattr(batch, "error", "") or "") or None
        return mapping.get(raw.split(".")[-1], BatchStatus.UNKNOWN), error

    def results(self, remote_job_id: str) -> list[BatchResult]:
        batch = self.client.batches.get(name=remote_job_id)
        output = getattr(batch, "dest", None) or getattr(batch, "output", None)
        file_name = getattr(output, "file_name", None) or getattr(output, "fileName", None)
        if not file_name:
            return []
        content = self.client.files.download(file=file_name)
        rows = []
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("key") or item.get("metadata", {}).get("key")
            rows.append(
                BatchResult(
                    custom_id=key,
                    response=item.get("response"),
                    error=item.get("error"),
                    usage=(item.get("response") or {}).get("usageMetadata")
                    or (item.get("response") or {}).get("usage_metadata"),
                    usage_scope=(
                        "request"
                        if (item.get("response") or {}).get("usageMetadata")
                        or (item.get("response") or {}).get("usage_metadata")
                        else None
                    ),
                )
            )
        return rows
