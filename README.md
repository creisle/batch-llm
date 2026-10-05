# batch-llm

Small, durable Python client for OpenAI and Gemini batch APIs. It uses local SQLite state, supports concurrent processes on one machine, and safely resumes jobs after process crashes.

## Install

```bash
pip install batch-llm
```

## Quick start

Set your API key:

```bash
export OPENAI_API_KEY=...
```

or:

```bash
export GEMINI_API_KEY=...
```

Then:

```python
from batch_llm import complete_prompts

results = complete_prompts(
    ["first prompt", "second prompt"],
    model="gpt-5.6",
    generation_config={"temperature": 0},
)

for row in results:
    print(row["generated_text"])
```

For Gemini:

```python
results = complete_prompts(
    ["first prompt", "second prompt"],
    model="gemini-2.5-pro",
    generation_config={"temperature": 0},
)
```

`complete_prompts()` returns `list[dict]` in the same order as the input prompts. Only unique, uncached prompt + model + configuration combinations create new provider requests. Duplicate prompts in the same call are submitted once, and if a later call overlaps previous work, only the missing requests are submitted.

For example, if one call completes 100 prompts and a later call contains 100 prompts of which 50 are already cached, only the remaining 50 are submitted.

## Inputs

The input can be a single string:

```python
results = complete_prompts(
    "hello",
    model="gpt-5.6",
    generation_config={"temperature": 0},
)
```

an iterable of strings:

```python
results = complete_prompts(
    ["first prompt", "second prompt"],
    model="gpt-5.6",
    generation_config={"temperature": 0},
)
```

or mappings containing the prompt field:

```python
results = complete_prompts(
    [
        {"id": "a", "prompt": "first prompt"},
        {"id": "b", "prompt": "second prompt"},
    ],
    model="gpt-5.6",
    generation_config={"temperature": 0},
)
```

Input mapping fields are preserved in the corresponding output, with `generated_text` and response metadata added. Duplicate prompts remain in their original positions in the returned results.

Use `dataset_text_field` for a different prompt field:

```python
results = complete_prompts(
    [{"id": 1, "text": "hello"}],
    model="gpt-5.6",
    generation_config=config,
    dataset_text_field="text",
)
```

## Generation config

`generation_config` can be a mapping:

```python
config = {
    "system_prompt": "Answer concisely.",
    "generation": {
        "temperature": 0,
    },
}
```

A mapping containing only generation options is shorthand for the `generation` object:

```python
{"temperature": 0}
```

A JSON config filename can also be passed directly:

```python
results = complete_prompts(
    ["hello"],
    model="gpt-5.6",
    generation_config="generation.json",
)
```

For example:

```json
{
  "system_prompt_file": "system.md",
  "generation": {
    "temperature": 0
  }
}
```

`system_prompt_file` follows these path rules:

- If `generation_config` is a filename, a relative `system_prompt_file` is resolved relative to that config file and an absolute path is used as-is.
- If `generation_config` is an in-memory mapping, `system_prompt_file` must be an absolute path. Use `system_prompt` for inline prompt text.
- `system_prompt` and `system_prompt_file` are mutually exclusive.

The contents of `system_prompt_file` are resolved before request/cache identity is calculated. The file path itself is not part of the cache key, so moving or renaming a prompt file without changing its contents does not invalidate cached results.

The cache identity includes the provider, model, resolved messages/system prompt, and generation configuration. Changing any model request parameter therefore creates a distinct cached request.

## Credentials

By default, credentials are read from:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

The provider is inferred from the model name unless specified otherwise.

For normal use, environment variables are recommended so API keys do not need to be embedded in source code.

A supported environment variable name can be selected explicitly:

```python
results = complete_prompts(
    ["hello"],
    model="gpt-5.6",
    generation_config=config,
    api_key_var="OPENAI_API_KEY",
)
```

For lower-level use, credentials can be passed directly to `BatchClient`:

```python
from batch_llm import BatchClient

client = BatchClient("openai", api_key="...")
# or
client = BatchClient("gemini", api_key="...")
```

## Storage

Batch state, cached responses, usage data, leases, and remote job identifiers are stored in SQLite.

By default, the database is placed in the OS user cache directory using `platformdirs`.

For example, on macOS:

```text
~/Library/Caches/batch-llm/state.sqlite3
```

On Linux, the normal platform cache location is used and `XDG_CACHE_HOME` is respected.

### Configure the cache location

Set a process/user-wide cache location with:

```bash
export BATCH_LLM_CACHE_DIR=/data/batch-llm-cache
```

which produces:

```text
/data/batch-llm-cache/state.sqlite3
```

Or override the location for an individual call:

```python
results = complete_prompts(
    ["hello"],
    model="gpt-5.6",
    generation_config=config,
    cache_dir="/data/batch-llm-cache",
)
```

`cache_dir` can also be an explicit SQLite filename:

```python
results = complete_prompts(
    ["hello"],
    model="gpt-5.6",
    generation_config=config,
    cache_dir="/data/my-project/llm-state.sqlite3",
)
```

Precedence is:

```text
cache_dir=
BATCH_LLM_CACHE_DIR
platformdirs OS cache location
```

The database should be stored on a local filesystem when multiple processes share it.

## Poll interval

The batch polling interval defaults to 30 seconds.

Set it process-wide with:

```bash
export BATCH_LLM_WAIT_INTERVAL_S=60
```

Or override it for one call:

```python
results = complete_prompts(
    ["hello"],
    model="gpt-5.6",
    generation_config=config,
    wait_interval_s=60,
)
```

An explicit function argument takes precedence over the environment variable.

## Results and usage

Each result contains the generated text, full provider response, request metadata, batch identifiers, and provider-native usage information.

For example:

```python
for row in results:
    print(row["generated_text"])
    print(row["usage"])
```

`usage` preserves the provider-native accounting payload unchanged:

- OpenAI: `usage`
- Gemini: `usageMetadata` / `usage_metadata`

Each newly executed provider request also has:

```python
{
    "usage_scope": "request",
    "usage_id": "...",
    "usage": {...},
}
```

`usage_id` identifies the exact provider request that incurred the usage.

If the same cached completion is reused for duplicate input rows, the usage information remains visible on each returned row, but the rows share the same `usage_id`.

When calculating token or cost totals, count each unique `usage_id` once.

The complete raw provider response is also retained so additional usage fields introduced by providers remain available.

Legacy caches migrated from an old multi-choice (`n > 1`) response cannot provide exact per-choice usage. Such usage is stored once with:

```text
usage_scope="legacy_aggregate"
```

rather than being incorrectly attributed to individual choices.

## Crash safety

SQLite uses WAL mode, short write transactions, expiring leases, and lease heartbeats during uploads and remote API calls.

Each exact provider/model/request/config combination has a deterministic local job identity. Repeating the same work resumes or reuses existing work instead of blindly submitting another batch.

Uploaded input-file IDs are also persisted and reused.

Before remote batch creation, the job is persisted as `submitting`. If the process dies after the provider accepts the request but before the remote job ID is saved, recovery attempts to locate the existing provider job using its deterministic marker.

If `batch-llm` cannot determine whether the provider accepted the submission, it raises `SubmissionUncertainError` rather than automatically resubmitting potentially expensive work.

This is intentionally conservative: avoiding an unnecessary duplicate batch takes priority over automatic resubmission when the remote state is uncertain.

## Concurrency

Multiple local processes can safely use the same SQLite database.

Concurrent callers coordinate through SQLite leases so the same missing work is not normally submitted more than once.

The database should remain on a local filesystem. If workers need to coordinate across multiple machines, use a client/server coordination system instead.

## Lower-level API

Most users should use `complete_prompts()`.

For direct access to the batch lifecycle:

```python
from batch_llm import BatchClient

client = BatchClient("openai")

job = client.submit(
    ["one", "two"],
    model="gpt-5.6",
)

job = client.wait(job)
results = client.results(job)
```

## Migrate an existing response cache

A one-off migration utility is included for SQLite caches using the `requests_cache_detailed` schema:

```bash
python scripts/migrate_legacy_cache.py /path/to/old-cache.sqlite
```

By default it imports into the configured `batch-llm` database. To target a specific database or cache directory:

```bash
python scripts/migrate_legacy_cache.py /path/to/old-cache.sqlite \
  --destination /path/to/state.sqlite3
```

The migration reports progress for large databases. It preserves repeated cached responses and usage data. Multi-choice responses are split into independent cached samples. Because old `n > 1` usage applies to the original API call rather than an individual choice, it is stored once with `usage_scope="legacy_aggregate"` and a stable `usage_id`.

## Development

Clone the repository and install the development environment with Poetry:

```bash
poetry install
```

Run the test suite:

```bash
poetry run pytest
```

Pytest is configured with branch coverage:

```toml
[tool.pytest.ini_options]
addopts = "--strict-markers --strict-config --cov=batch_llm --cov-report=term-missing --cov-branch"
```

Run Ruff:

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
```

The normal test suite uses fake provider clients and does not make paid API calls.
