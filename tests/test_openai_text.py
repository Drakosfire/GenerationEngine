"""OpenAI adapter: SDK retry ownership and request vs response IDs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from generationengine.providers.base import TextGenerationCall
from generationengine.providers.openai_text import OpenAITextProvider, _ids_from_response


@pytest.mark.asyncio
async def test_openai_adapter_closes_underlying_async_sdk_client_once_per_call() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    client = FakeClient()
    provider = OpenAITextProvider(client=client)

    await provider.aclose()

    assert client.close_calls == 1


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
            max_output_tokens=400,
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
    assert [call["max_output_tokens"] for call in responses.calls] == [400, 400]
    assert responses.calls[0]["text"]["format"]["type"] == "json_schema"
    assert "did not satisfy the required schema" in responses.calls[1]["input"]


def _openai_call(**overrides) -> TextGenerationCall:
    payload: dict = {"model": "gpt-5.1", "user_prompt": "hello"}
    payload.update(overrides)
    return TextGenerationCall(**payload)


def _openai_kwargs(call: TextGenerationCall, *, streaming: bool = False) -> dict:
    provider = OpenAITextProvider(client=SimpleNamespace())
    return provider._request_kwargs(call, streaming=streaming)


def test_openai_omitted_temperature_forwards_0_7() -> None:
    kwargs = _openai_kwargs(_openai_call())
    assert kwargs["temperature"] == 0.7


def test_openai_numeric_temperature_is_forwarded_exactly() -> None:
    assert _openai_kwargs(_openai_call(temperature=0.2))["temperature"] == 0.2


def test_openai_zero_temperature_is_forwarded() -> None:
    assert _openai_kwargs(_openai_call(temperature=0.0))["temperature"] == 0.0


def test_openai_none_temperature_omits_provider_field() -> None:
    kwargs = _openai_kwargs(_openai_call(temperature=None))
    assert "temperature" not in kwargs


def test_openai_stream_none_temperature_omits_provider_field() -> None:
    kwargs = _openai_kwargs(_openai_call(temperature=None), streaming=True)
    assert "temperature" not in kwargs


def test_openai_stream_numeric_temperature_is_forwarded() -> None:
    assert _openai_kwargs(_openai_call(temperature=0.2), streaming=True)["temperature"] == 0.2


def test_openai_none_output_ceiling_omits_provider_field() -> None:
    kwargs = _openai_kwargs(_openai_call(max_output_tokens=None))
    assert "max_output_tokens" not in kwargs


def test_openai_output_ceiling_is_forwarded_exactly() -> None:
    assert _openai_kwargs(_openai_call(max_output_tokens=400))["max_output_tokens"] == 400


def test_openai_stream_output_ceiling_is_forwarded_exactly() -> None:
    kwargs = _openai_kwargs(_openai_call(max_output_tokens=400), streaming=True)
    assert kwargs["max_output_tokens"] == 400


def test_openai_false_json_object_omits_format() -> None:
    kwargs = _openai_kwargs(_openai_call(json_object=False))
    assert "text" not in kwargs


def test_openai_json_object_maps_to_responses_format_and_composes_controls() -> None:
    kwargs = _openai_kwargs(
        _openai_call(
            json_object=True,
            temperature=0.2,
            max_output_tokens=400,
        )
    )
    assert kwargs["text"] == {"format": {"type": "json_object"}}
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_output_tokens"] == 400


@pytest.mark.asyncio
async def test_openai_json_object_success_remains_raw_text() -> None:
    responses = _FakeResponses([_sdk_response(text="{not-json")])
    provider = OpenAITextProvider(client=SimpleNamespace(responses=responses))
    result = await provider.generate(_openai_call(json_object=True))
    assert responses.calls[0]["text"] == {"format": {"type": "json_object"}}
    assert result.text == "{not-json"
    assert result.parsed is None


@pytest.mark.asyncio
async def test_openai_generate_omits_temperature_when_none() -> None:
    responses = _FakeResponses([_sdk_response(text="ok")])
    provider = OpenAITextProvider(client=SimpleNamespace(responses=responses))
    await provider.generate(_openai_call(temperature=None))
    assert "temperature" not in responses.calls[0]
    assert responses.calls[0]["model"] == "gpt-5.1"
