import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

try:
    from google import genai as _genai  # noqa: F401
except ImportError:
    google_mod = sys.modules.setdefault("google", ModuleType("google"))
    genai_mod = ModuleType("google.genai")
    types_mod = ModuleType("google.genai.types")

    class UploadFileConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    types_mod.UploadFileConfig = UploadFileConfig
    genai_mod.types = types_mod
    genai_mod.Client = object
    google_mod.genai = genai_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod


try:
    import openai as _openai  # noqa: F401
except ImportError:
    openai_mod = ModuleType("openai")
    openai_mod.OpenAI = object
    sys.modules["openai"] = openai_mod

from batch_llm.models import BatchStatus
from batch_llm.providers.gemini import GeminiProvider
from batch_llm.providers.openai import OpenAIProvider


class OpenAIFiles:
    def __init__(self, result_content=b""):
        self.result_content = result_content
        self.created = []

    def create(self, *, file, purpose):
        self.created.append((file.read(), purpose))
        return SimpleNamespace(id="file-1")

    def content(self, file_id):
        return SimpleNamespace(content=self.result_content)


class OpenAIBatches:
    def __init__(self):
        self.created = []
        self.jobs = {}
        self.listed = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id="batch-1")

    def list(self, limit=100):
        return list(self.listed)

    def retrieve(self, job_id):
        return self.jobs[job_id]


class OpenAIClient:
    def __init__(self, result_content=b""):
        self.files = OpenAIFiles(result_content)
        self.batches = OpenAIBatches()


class GeminiFiles:
    def __init__(self, result_content=b""):
        self.result_content = result_content
        self.uploads = []

    def upload(self, *, file, config):
        self.uploads.append((Path(file), config))
        return SimpleNamespace(name="files/input")

    def download(self, *, file):
        return self.result_content


class GeminiBatches:
    def __init__(self):
        self.created = []
        self.jobs = {}
        self.listed = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(name="batches/1")

    def list(self):
        return list(self.listed)

    def get(self, *, name):
        return self.jobs[name]


class GeminiClient:
    def __init__(self, result_content=b""):
        self.files = GeminiFiles(result_content)
        self.batches = GeminiBatches()


def test_openai_write_input_contract(tmp_path: Path):
    provider = OpenAIProvider(client=OpenAIClient())
    path = tmp_path / "input.jsonl"
    provider.write_input(
        [("request-0", {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]})],
        path,
    )
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row == {
        "custom_id": "request-0",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
    }


def test_openai_create_uses_deterministic_metadata():
    client = OpenAIClient()
    provider = OpenAIProvider(client=client)
    assert provider.create(job_id="local-1", model="ignored", file_id="file-1") == "batch-1"
    assert client.batches.created == [
        {
            "input_file_id": "file-1",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"batch_llm_job": "local-1"},
        }
    ]


def test_openai_find_created_job_and_status_mapping():
    client = OpenAIClient()
    client.batches.listed = [
        SimpleNamespace(id="other", metadata={}),
        SimpleNamespace(id="wanted", metadata={"batch_llm_job": "local-1"}),
    ]
    client.batches.jobs["wanted"] = SimpleNamespace(status="in_progress", errors=None)
    provider = OpenAIProvider(client=client)
    assert provider.find_created_job(job_id="local-1") == "wanted"
    assert provider.status("wanted") == (BatchStatus.RUNNING, None)


def test_openai_results_preserve_per_request_usage():
    payload = b"\n".join(
        [
            json.dumps(
                {
                    "custom_id": "request-1",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "B"}}],
                            "usage": {
                                "prompt_tokens": 5,
                                "completion_tokens": 2,
                                "total_tokens": 7,
                            },
                        }
                    },
                    "error": None,
                }
            ).encode(),
            json.dumps(
                {
                    "custom_id": "request-0",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "A"}}],
                            "usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 1,
                                "total_tokens": 5,
                            },
                        }
                    },
                    "error": None,
                }
            ).encode(),
        ]
    )
    client = OpenAIClient(payload)
    client.batches.jobs["batch"] = SimpleNamespace(output_file_id="output")
    rows = OpenAIProvider(client=client).results("batch")
    assert [row.custom_id for row in rows] == ["request-1", "request-0"]
    assert [row.usage["total_tokens"] for row in rows] == [7, 5]
    assert all(row.usage_scope == "request" for row in rows)


def test_gemini_request_translation_preserves_system_and_generation():
    request = GeminiProvider._request(
        {
            "model": "gemini-test",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "max_tokens": 50,
            "top_p": 0.9,
            "stop": ["END"],
        }
    )
    assert request["system_instruction"] == {"parts": [{"text": "system"}]}
    assert request["contents"] == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},
    ]
    assert request["generation_config"] == {
        "maxOutputTokens": 50,
        "topP": 0.9,
        "stopSequences": ["END"],
    }


def test_gemini_create_and_reconcile_marker():
    client = GeminiClient()
    provider = GeminiProvider(client=client)
    assert provider.create(job_id="abc", model="gemini-test", file_id="files/input") == "batches/1"
    assert client.batches.created == [
        {"model": "gemini-test", "src": "files/input", "config": {"display_name": "batch-llm-abc"}}
    ]
    client.batches.listed = [SimpleNamespace(name="batches/2", display_name="batch-llm-abc")]
    assert provider.find_created_job(job_id="abc") == "batches/2"


def test_gemini_results_preserve_per_request_usage():
    payload = json.dumps(
        {
            "key": "request-0",
            "response": {
                "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 5,
                },
            },
        }
    ).encode()
    client = GeminiClient(payload)
    client.batches.jobs["batches/1"] = SimpleNamespace(dest=SimpleNamespace(file_name="files/out"))
    row = GeminiProvider(client=client).results("batches/1")[0]
    assert row.custom_id == "request-0"
    assert row.usage == {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}
    assert row.usage_scope == "request"


def test_openai_upload_and_no_matching_job(tmp_path: Path):
    client = OpenAIClient()
    provider = OpenAIProvider(client=client)
    path = tmp_path / "input.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    assert provider.upload(path) == "file-1"
    assert client.files.created[0][1] == "batch"
    client.batches.listed = [SimpleNamespace(id="x", metadata={"batch_llm_job": "other"})]
    assert provider.find_created_job(job_id="wanted") is None


def test_openai_status_error_unknown_and_results_without_output():
    client = OpenAIClient()
    provider = OpenAIProvider(client=client)
    client.batches.jobs["failed"] = SimpleNamespace(status="failed", errors={"message": "bad"})
    status, error = provider.status("failed")
    assert status == BatchStatus.FAILED
    assert "bad" in error
    client.batches.jobs["unknown"] = SimpleNamespace(status="new_state", errors=None)
    assert provider.status("unknown") == (BatchStatus.UNKNOWN, None)
    client.batches.jobs["empty"] = SimpleNamespace(output_file_id=None)
    assert provider.results("empty") == []


def test_gemini_write_input_and_upload(tmp_path: Path):
    client = GeminiClient()
    provider = GeminiProvider(client=client)
    path = tmp_path / "input.jsonl"
    provider.write_input(
        [("request-0", {"model": "gemini-test", "messages": [{"role": "user", "content": "hi"}]})],
        path,
    )
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["key"] == "request-0"
    assert row["request"]["contents"][0]["parts"] == [{"text": "hi"}]
    assert provider.upload(path) == "files/input"
    assert client.files.uploads[0][0] == path


def test_gemini_request_rejects_nontext_content():
    try:
        GeminiProvider._request(
            {"model": "m", "messages": [{"role": "user", "content": [{"text": "hi"}]}]}
        )
    except ValueError as exc:
        assert "text message content only" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_gemini_status_mappings_and_no_output():
    client = GeminiClient()
    provider = GeminiProvider(client=client)
    client.batches.jobs["running"] = SimpleNamespace(state="JOB_STATE_RUNNING", error=None)
    assert provider.status("running") == (BatchStatus.RUNNING, None)
    client.batches.jobs["failed"] = SimpleNamespace(state="JOB_STATE_FAILED", error="bad")
    assert provider.status("failed") == (BatchStatus.FAILED, "bad")
    client.batches.jobs["unknown"] = SimpleNamespace(state="JOB_STATE_NEW", error=None)
    assert provider.status("unknown") == (BatchStatus.UNKNOWN, None)
    client.batches.jobs["empty"] = SimpleNamespace(dest=None, output=None)
    assert provider.results("empty") == []


def test_openai_results_include_error_file_rows():
    output = json.dumps(
        {
            "custom_id": "request-0",
            "response": {"body": {"choices": [{"message": {"content": "ok"}}]}},
            "error": None,
        }
    ).encode()
    error = json.dumps(
        {
            "custom_id": "request-1",
            "response": None,
            "error": {"code": "bad_request", "message": "invalid request"},
        }
    ).encode()

    class Files:
        def content(self, file_id):
            return SimpleNamespace(content={"output": output, "errors": error}[file_id])

    client = OpenAIClient()
    client.files = Files()
    client.batches.jobs["batch"] = SimpleNamespace(output_file_id="output", error_file_id="errors")
    rows = OpenAIProvider(client=client).results("batch")
    assert [row.custom_id for row in rows] == ["request-0", "request-1"]
    assert rows[0].response["choices"][0]["message"]["content"] == "ok"
    assert rows[1].response is None
    assert rows[1].error == {"code": "bad_request", "message": "invalid request"}


def test_openai_results_surface_error_from_response_body():
    payload = json.dumps(
        {
            "custom_id": "request-0",
            "response": {
                "status_code": 400,
                "request_id": "req_123",
                "body": {
                    "error": {
                        "message": "Unsupported parameter",
                        "type": "invalid_request_error",
                        "code": "unsupported_parameter",
                    }
                },
            },
            "error": None,
        }
    ).encode()
    client = OpenAIClient(payload)
    client.batches.jobs["batch-1"] = SimpleNamespace(output_file_id="output", error_file_id=None)
    provider = OpenAIProvider(client=client)
    rows = provider.results("batch-1")
    assert len(rows) == 1
    assert rows[0].error == {
        "message": "Unsupported parameter",
        "type": "invalid_request_error",
        "code": "unsupported_parameter",
    }


def test_openai_request_filters_local_generation_options_and_maps_max_new_tokens(tmp_path: Path):
    provider = OpenAIProvider(client=OpenAIClient(b""))
    path = tmp_path / "input.jsonl"
    provider.write_input(
        [
            (
                "request-0",
                {
                    "model": "gpt-test",
                    "messages": [{"role": "user", "content": "hello"}],
                    "min_new_tokens": 5,
                    "max_new_tokens": 100,
                    "do_sample": False,
                    "use_cache": True,
                    "truncation": True,
                    "temperature": 0,
                },
            )
        ],
        path,
    )
    body = json.loads(path.read_text().strip())["body"]
    assert body == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 100,
        "temperature": 0,
    }


def test_gemini_request_filters_local_generation_options_and_maps_max_new_tokens():
    request = GeminiProvider._request(
        {
            "model": "gemini-test",
            "messages": [{"role": "user", "content": "hello"}],
            "min_new_tokens": 5,
            "max_new_tokens": 100,
            "do_sample": False,
            "use_cache": True,
            "truncation": True,
            "temperature": 0,
        }
    )
    assert request["generation_config"] == {"maxOutputTokens": 100, "temperature": 0}
