from __future__ import annotations

from typing import Any

_IGNORED_GENERATION_KEYS = {"do_sample", "use_cache", "truncation", "min_new_tokens"}


def ignored_generation_keys(body: dict[str, Any]) -> set[str]:
    """Return generation keys that are intentionally omitted from provider requests."""
    return _IGNORED_GENERATION_KEYS.intersection(body)


def openai_request_body(body: dict[str, Any]) -> dict[str, Any]:
    """Return the effective OpenAI request body used for submission and identity."""
    aliases = {"max_new_tokens": "max_completion_tokens"}
    result: dict[str, Any] = {}
    for key, value in body.items():
        if key in _IGNORED_GENERATION_KEYS:
            continue
        mapped = aliases.get(key, key)
        if mapped in result and mapped != key:
            raise ValueError(f"generation options {key!r} and {mapped!r} both map to {mapped!r}")
        result[mapped] = value
    return result


def gemini_request_body(body: dict[str, Any]) -> dict[str, Any]:
    """Return the effective native Gemini GenerateContent request body."""
    messages = body.get("messages", [])
    contents = []
    system_parts = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            parts = [{"text": content}]
        else:
            raise ValueError("Gemini adapter currently supports text message content only")
        if role in {"system", "developer"}:
            system_parts.extend(parts)
        else:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})

    generation = {
        k: v
        for k, v in body.items()
        if k not in {"model", "messages", "n"} and k not in _IGNORED_GENERATION_KEYS
    }
    aliases = {
        "max_tokens": "maxOutputTokens",
        "max_completion_tokens": "maxOutputTokens",
        "max_new_tokens": "maxOutputTokens",
        "top_p": "topP",
        "top_k": "topK",
        "stop": "stopSequences",
    }
    translated: dict[str, Any] = {}
    for key, value in generation.items():
        mapped = aliases.get(key, key)
        if mapped in translated:
            raise ValueError(f"multiple generation options map to Gemini parameter {mapped!r}")
        translated[mapped] = value

    request: dict[str, Any] = {"contents": contents}
    if translated:
        request["generation_config"] = translated
    if system_parts:
        request["system_instruction"] = {"parts": system_parts}
    return request


def effective_request_body(provider: str, body: dict[str, Any]) -> dict[str, Any]:
    provider = provider.lower()
    if provider == "openai":
        return openai_request_body(body)
    if provider == "gemini":
        return gemini_request_body(body)
    raise ValueError(f"unsupported provider: {provider!r}")
