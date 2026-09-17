"""OpenAI adapter: SDK retry ownership and request vs response IDs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from generationengine.providers.openai_text import OpenAITextProvider, _ids_from_response


def test_openai_client_disables_sdk_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "generationengine.providers.openai_text.AsyncOpenAI",
        FakeAsyncOpenAI,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    OpenAITextProvider()
    assert captured["api_key"] == "sk-test"
    assert captured["max_retries"] == 0


def test_request_id_is_http_id_not_response_object_id() -> None:
    response = SimpleNamespace(
        id="resp_abc",
        _request_id="req_http",
        output_text="ok",
        model="gpt-5.1",
        usage=None,
        refusal=None,
    )
    request_id, response_id = _ids_from_response(response)
    assert request_id == "req_http"
    assert response_id == "resp_abc"
    result = OpenAITextProvider(client=SimpleNamespace())._result_from_response(response)
    assert result.provider_request_id == "req_http"
    assert result.provider_response_id == "resp_abc"


def test_openai_maps_sdk_errors_to_safe_public_messages() -> None:
    from generationengine.failures import FailureCode

    provider = OpenAITextProvider(client=SimpleNamespace())
    error = provider._map_exception(
        RuntimeError("Authorization Bearer sk-live HTTP 502 from api.openai.com")
    )
    assert error.failure.code is FailureCode.PROVIDER_ERROR
    assert error.failure.message == "Provider request failed."
    assert "sk-live" not in error.failure.message
    assert "openai.com" not in error.failure.message

    class FakeTimeoutError(Exception):
        pass

    timeout = provider._map_exception(FakeTimeoutError("waited 45s"))
    assert timeout.failure.code is FailureCode.PROVIDER_TIMEOUT
    assert timeout.failure.message == "Provider request timed out."

    class FakeRateLimitError(Exception):
        pass

    rate_limited = provider._map_exception(
        FakeRateLimitError(
            "429 https://api.openai.com/v1/responses Authorization Bearer sk-live"
        )
    )
    assert rate_limited.failure.code is FailureCode.RATE_LIMITED
    assert rate_limited.failure.message == "Provider rate limit exceeded."
    assert "sk-live" not in rate_limited.failure.message
    assert "openai.com" not in rate_limited.failure.message


def _sdk_response(*, text: str, input_tokens: int = 4, output_tokens: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_abc",
        _request_id="req_http",
        output_text=text,
        model="gpt-5.1",
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        refusal=None,
    )


class _FakeResponses:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected extra OpenAI SDK call")
        return self.responses.pop(0)


def test_openai_malformed_structured_json_returns_raw_text() -> None:
    response = _sdk_response(text="{not-json")
    result = OpenAITextProvider(client=SimpleNamespace())._result_from_response(response)
    assert result.text == "{not-json"
    assert result.parsed is None
    assert result.input_tokens == 4
    assert result.output_tokens == 1


@pytest.mark.asyncio
async def test_openai_adapter_malformed_json_is_repaired_by_conformance() -> None:
    from generationengine import GenerationClient, InferenceProfile, TextRequest

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name", "count"],
        "additionalProperties": False,
    }
    responses = _FakeResponses(
        [
            _sdk_response(text="{not-json", input_tokens=7, output_tokens=1),
            _sdk_response(text='{"name":"ok","count":1}', input_tokens=9, output_tokens=2),
        ]
    )
    client = GenerationClient(
        text_provider=OpenAITextProvider(client=SimpleNamespace(responses=responses))
    )
    result = await client.generate_structured(
        TextRequest(
            user_prompt="make fixture",
            profile=InferenceProfile.STRUCTURED_LOW_COST,
            json_schema=schema,
            schema_name="fixture",
        )
    )
    assert result.parsed == {"name": "ok", "count": 1}
    assert result.observation.conformance_retry_count == 1
    assert result.observation.retry_count == 0
    assert result.observation.transport_retry_count == 0
    assert result.observation.provider_attempt_count == 2
    assert result.observation.input_tokens == 16
    assert result.observation.output_tokens == 3
    assert len(responses.calls) == 2
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"
    assert "did not satisfy the required schema" in responses.calls[1]["input"]
