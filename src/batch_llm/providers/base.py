from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import BatchResult, BatchStatus


class Provider(ABC):
    name: str

    def cache_identity_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return the effective provider request body used for cache/job identity."""
        return body

    @abstractmethod
    def write_input(self, requests: list[tuple[str, dict[str, Any]]], path: Path) -> None:
        ...

    @abstractmethod
    def upload(self, path: Path) -> str:
        ...

    @abstractmethod
    def create(self, *, job_id: str, model: str, file_id: str) -> str:
        ...

    @abstractmethod
    def find_created_job(self, *, job_id: str) -> str | None:
        """Find a remotely created job using the package's deterministic marker."""
        ...

    @abstractmethod
    def status(self, remote_job_id: str) -> tuple[BatchStatus, str | None]:
        ...

    @abstractmethod
    def results(self, remote_job_id: str) -> list[BatchResult]:
        ...
