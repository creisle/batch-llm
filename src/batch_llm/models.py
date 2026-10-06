from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class BatchStatus(str, Enum):
    PREPARING = "preparing"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED, self.EXPIRED}


@dataclass(frozen=True)
class BatchJob:
    id: str
    provider: str
    model: str
    status: BatchStatus
    remote_file_id: str | None = None
    remote_job_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == BatchStatus.COMPLETED


@dataclass(frozen=True)
class BatchResult:
    custom_id: str
    response: dict[str, Any] | None
    error: dict[str, Any] | str | None = None
    usage: dict[str, Any] | None = None
    usage_scope: str | None = None
    usage_id: str | None = None
