from __future__ import annotations

import json
import logging
import os
import time
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from .client import BatchClient
from .cache import request_cache_hash
from .request_identity import effective_request_body, ignored_generation_keys

logger = logging.getLogger(__name__)

GenerationConfig = str | os.PathLike[str] | Mapping[str, Any]
PromptInput = str | Iterable[str] | Iterable[Mapping[str, Any]]
_RESERVED_CONFIG_KEYS = {"generation", "system_prompt", "system_prompt_file"}


def load_generation_config(config: GenerationConfig) -> dict[str, Any]:
    """Load generation options from a JSON file or mapping.

    Accepted mapping forms::

        {"generation": {...}, "system_prompt": "..."}
        {"generation": {...}, "system_prompt_file": "/absolute/system.md"}
        {"system_prompt": "...", "temperature": 0}
        {"temperature": 0, ...}

    Relative ``system_prompt_file`` paths are resolved relative to the JSON
    config file. For an in-memory mapping, ``system_prompt_file`` must be an
    absolute path.
    """
    config_from_file = isinstance(config, (str, os.PathLike))
    if config_from_file:
        path = Path(config).expanduser().resolve()
        config_dir = path.parent
        with path.open("r", encoding="utf-8") as fh:
            loaded: dict[str, Any] = json.load(fh)
    else:
        config_dir = None
        loaded = dict(config)

    has_reserved = bool(_RESERVED_CONFIG_KEYS.intersection(loaded))
    if not has_reserved:
        loaded = {"generation": loaded}
    else:
        top_level_generation = {k: v for k, v in loaded.items() if k not in _RESERVED_CONFIG_KEYS}
        config_values = {k: v for k, v in loaded.items() if k in _RESERVED_CONFIG_KEYS}
        nested_generation = config_values.get("generation")
        if nested_generation is None:
            nested_generation = {}
        elif not isinstance(nested_generation, Mapping):
            raise TypeError("generation must be a mapping")
        else:
            nested_generation = dict(nested_generation)

        conflicts = set(top_level_generation).intersection(nested_generation)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                f"generation option(s) specified both at top level and inside 'generation': {names}"
            )

        config_values["generation"] = {**nested_generation, **top_level_generation}
        loaded = config_values

    loaded = dict(loaded)
    generation = loaded.get("generation")
    if generation is None:
        loaded["generation"] = {}
    elif not isinstance(generation, Mapping):
        raise TypeError("generation must be a mapping")
    else:
        loaded["generation"] = dict(generation)

    system_prompt = loaded.get("system_prompt")
    prompt_file = loaded.get("system_prompt_file")
    if system_prompt is not None and prompt_file is not None:
        raise ValueError("system_prompt and system_prompt_file are mutually exclusive")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string")

    if prompt_file is not None:
        if not isinstance(prompt_file, (str, os.PathLike)):
            raise TypeError("system_prompt_file must be a path")
        prompt_path = Path(prompt_file).expanduser()
        if not prompt_path.is_absolute():
            if not config_from_file:
                raise ValueError(
                    "system_prompt_file must be an absolute path when generation_config "
                    "is provided as a mapping"
                )
            prompt_path = config_dir / prompt_path
        prompt_path = prompt_path.resolve()
        with prompt_path.open("r", encoding="utf-8") as fh:
            loaded["system_prompt"] = fh.read().strip()
        loaded["system_prompt_file"] = str(prompt_path)

    return loaded


def complete_prompts(
    prompts: PromptInput,
    model: str,
    generation_config: GenerationConfig,
    *,
    api_key_var: Literal["GEMINI_API_KEY", "OPENAI_API_KEY"] | None = None,
    dataset_text_field: str = "prompt",
    min_repeat: int = 1,
    wait_interval_s: float | None = None,
    total_wait_s: float | None = 30 * 15,
    cache_dir: str | os.PathLike[str] | None = None,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """Complete prompts using the provider batch API.

    ``prompts`` may be a single string, an iterable of strings, or an iterable
    of mappings containing ``dataset_text_field``. Mapping fields are copied to
    the returned rows.

    ``generation_config`` may be a JSON filename or a mapping.

    Identical work is persisted in SQLite and resumed across process restarts.
    A submission whose remote state cannot be proven after a crash is never
    automatically resubmitted.
    """
    if min_repeat < 1:
        raise ValueError("min_repeat must be >= 1")

    if wait_interval_s is None:
        wait_interval_s = _env_float("BATCH_LLM_WAIT_INTERVAL_S", 30.0)

    source = _normalize_inputs(prompts, dataset_text_field)
    if not source:
        return []

    cfg = load_generation_config(generation_config)
    generation = dict(cfg.get("generation") or {})
    system_prompt = cfg.get("system_prompt")

    ignored = sorted(ignored_generation_keys(generation))
    if ignored:
        warnings.warn(
            "batch-llm will ignore generation parameter(s) that are not sent to the provider "
            f"and do not affect cache identity: {', '.join(ignored)}",
            UserWarning,
            stacklevel=2,
        )

    if provider is None:
        if api_key_var:
            provider = "gemini" if api_key_var == "GEMINI_API_KEY" else "openai"
        else:
            provider = "gemini" if "gemini" in model.lower() else "openai"
    provider = provider.lower()
    if provider not in {"openai", "gemini"}:
        raise ValueError("provider must be 'openai' or 'gemini'")

    if api_key_var is None:
        api_key_var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    expected = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    if api_key_var != expected:
        raise ValueError(f"{provider} requires {expected}, got {api_key_var}")

    client = BatchClient(provider, api_key=os.environ.get(api_key_var), storage_path=cache_dir)

    unique_prompts = list(dict.fromkeys(row[dataset_text_field] for row in source))

    def request_body(prompt: str) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {"model": model, "messages": messages, **generation}

    by_prompt: dict[str, list[Any]] = {}
    missing_prompts: list[str] = []
    missing_keys: list[str] = []
    for prompt in unique_prompts:
        effective_body = effective_request_body(provider, request_body(prompt))
        cache_key = request_cache_hash(provider, effective_body)
        client.store.ensure_cache_request(cache_key, provider, effective_body)
        cached = client.store.cached_responses(cache_key)[:min_repeat]
        by_prompt[prompt] = list(cached)
        for _ in range(min_repeat - len(cached)):
            missing_prompts.append(prompt)
            missing_keys.append(cache_key)

    job = None
    if missing_prompts:
        job = client.submit(
            missing_prompts, model=model, generation=generation, system_prompt=system_prompt
        )
        wait_started = time.monotonic()
        job = client.wait(job, poll_interval=wait_interval_s, timeout=total_wait_s)
        if not job.completed:
            detail = f": {job.error}" if job.error else ""
            raise RuntimeError(
                f"{provider} batch {job.id} ({job.remote_job_id or 'no remote id'}) "
                f"ended with status={job.status.value}{detail}"
            )

        # Providers can briefly report a batch as completed before its output
        # file is readable. Keep checking within the original wait budget
        # rather than treating a transient empty/partial result set as final.
        while True:
            batch_results = client.results(job, refresh=False)
            if len(batch_results) == len(missing_prompts):
                break
            elapsed = time.monotonic() - wait_started
            if total_wait_s is not None and elapsed >= total_wait_s:
                errors = [
                    _format_provider_error(result.error)
                    for result in batch_results
                    if result.error is not None
                ]
                error_detail = f"; provider errors: {' | '.join(errors[:5])}" if errors else ""
                raise RuntimeError(
                    f"{provider} batch {job.id} ({job.remote_job_id or 'no remote id'}) "
                    f"returned {len(batch_results)} results for {len(missing_prompts)} requests "
                    f"after waiting for results{error_detail}"
                )
            time.sleep(wait_interval_s)

        for prompt, cache_key, result in zip(missing_prompts, missing_keys, batch_results):
            by_prompt[prompt].append(result)
            if result.error is not None:
                logger.error(
                    "%s batch request failed: batch=%s remote_batch=%s request=%s error=%s",
                    provider,
                    job.id,
                    job.remote_job_id,
                    result.custom_id,
                    _format_provider_error(result.error),
                )
            if result.response is not None and result.error is None:
                client.store.add_cached_response(
                    cache_key,
                    result.response,
                    result.usage or _usage(result.response, provider),
                    usage_scope=result.usage_scope or ("request" if result.usage else None),
                    usage_id=result.usage_id
                    or (f"{job.id}:{result.custom_id}" if result.usage is not None else None),
                )

    config_json = json.dumps(cfg, sort_keys=True, default=str)
    output: list[dict[str, Any]] = []
    for source_row in source:
        prompt = source_row[dataset_text_field]
        for repeat_index, result in enumerate(by_prompt[prompt]):
            row = dict(source_row)
            row.update(
                {
                    "repeat_index": repeat_index,
                    "response": result.response,
                    "generated_text": _generated_text(result.response, provider),
                    "error": result.error,
                    "error_message": _format_provider_error(result.error)
                    if result.error is not None
                    else None,
                    "usage": result.usage or _usage(result.response, provider),
                    "usage_scope": result.usage_scope
                    or ("request" if (result.usage or _usage(result.response, provider)) else None),
                    "usage_id": result.usage_id,
                    "request.model": model,
                    "request.api": provider.upper(),
                    "config": config_json,
                    "batch_job_id": job.id if job else None,
                    "remote_batch_job_id": job.remote_job_id if job else None,
                }
            )
            for field, value in generation.items():
                row[f"request.{field}"] = value
            output.append(row)

    return output


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return parsed


def _normalize_inputs(prompts: PromptInput, dataset_text_field: str) -> list[dict[str, Any]]:
    if isinstance(prompts, str):
        items: list[Any] = [prompts]
    else:
        items = list(prompts)

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            row = {dataset_text_field: item}
        elif isinstance(item, Mapping):
            row = dict(item)
            if dataset_text_field not in row:
                raise KeyError(f"item {i} is missing dataset text field: {dataset_text_field}")
        else:
            raise TypeError(f"item {i} must be a string or mapping, got {type(item).__name__}")

        prompt = row[dataset_text_field]
        if not isinstance(prompt, str):
            raise TypeError(f"item {i} field {dataset_text_field!r} must be a string")
        rows.append(row)
    return rows


def _generated_text(response: dict[str, Any] | None, provider: str) -> str | None:
    if not response:
        return None

    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass

    try:
        parts = response["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts if isinstance(part, dict)) or None
    except (KeyError, IndexError, TypeError):
        return None


def _usage(response: dict[str, Any] | None, provider: str) -> dict[str, Any] | None:
    if not response:
        return None
    usage = response.get("usage") or response.get("usageMetadata") or response.get("usage_metadata")
    return dict(usage) if isinstance(usage, Mapping) else None


def _format_provider_error(error: Any) -> str:
    """Return a concise human-readable provider error without losing raw error data."""
    if error is None:
        return ""
    if isinstance(error, str):
        return error
    if isinstance(error, Mapping):
        nested = error.get("error")
        if isinstance(nested, Mapping):
            return _format_provider_error(nested)

        parts: list[str] = []
        for key in ("code", "type", "status", "message"):
            value = error.get(key)
            if value not in (None, ""):
                parts.append(f"{key}={value}")
        if parts:
            return ", ".join(parts)
        try:
            return json.dumps(dict(error), sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(error)
    return str(error)
