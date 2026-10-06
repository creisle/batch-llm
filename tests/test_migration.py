import importlib.util
import json
import sqlite3
from pathlib import Path

from batch_llm.cache import request_cache_hash
from batch_llm.storage import Store


def load_migration_module():
    path = Path(__file__).parents[1] / "scripts" / "migrate_legacy_cache.py"
    spec = importlib.util.spec_from_file_location("migrate_legacy_cache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_migrate_legacy_cache_splits_choices_without_double_counting_usage(tmp_path: Path):
    legacy = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(legacy)
    conn.execute(
        """
        CREATE TABLE requests_cache_detailed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request VARCHAR NOT NULL,
            response VARCHAR,
            last_updated datetime default current_timestamp
        )
        """
    )
    request = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "client.base_url": "https://api.openai.com/v1/",
    }
    response = {
        "choices": [{"message": {"content": "one"}}, {"message": {"content": "two"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    }
    conn.execute(
        "INSERT INTO requests_cache_detailed(request, response) VALUES (?, ?)",
        (json.dumps(request, sort_keys=True), json.dumps(response, sort_keys=True)),
    )
    conn.commit()
    conn.close()

    destination = tmp_path / "new.sqlite3"
    migrate = load_migration_module()
    assert migrate.migrate(legacy, destination) == (2, 0)

    body = dict(request)
    body.pop("client.base_url")
    key = request_cache_hash("openai", body)
    rows = Store(destination).cached_responses(key)
    assert [r.response["choices"][0]["message"]["content"] for r in rows] == ["one", "two"]
    assert rows[0].usage["total_tokens"] == 9
    assert rows[0].usage_scope == "legacy_aggregate"
    assert rows[0].usage_id is not None and rows[0].usage_id.startswith("legacy:")
    assert rows[1].usage is None
    assert rows[1].usage_scope is None
    assert rows[1].usage_id is None
