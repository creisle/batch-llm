from __future__ import annotations

import hashlib
import json
from typing import Any


def request_cache_hash(provider: str, body: dict[str, Any]) -> str:
    payload = {"provider": provider.lower(), "body": body}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
