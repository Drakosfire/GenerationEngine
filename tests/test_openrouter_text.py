"""OpenRouter adapter: chat completions transport with openrouter identity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from generationengine.failures import FailureCode
from generationengine.observation import ObservationState
from generationengine.providers.base import TextCompleted, TextDelta, TextFailed, TextGenerationCall
from generationengine.providers.errors import ProviderError
from generationengine.providers.openrouter_text import (
    OPENROUTER_BASE_URL,
    OpenRouterTextProvider,
    _ids_from_response,
)


def _chat_response(
    *,
    text: str = "ok",
    refusal: str | None = None,
    usage: object | None = SimpleNamespace(
        prompt_tokens=3,
        completion_tokens=2,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1),
    ),
) -> SimpleNamespace:
    return SimpleNamespace(
        id="chatcmpl-or",
        _request_id="req-or",
        model="deepseek/deepseek-v4.1-flash",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, refusal=refusal),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


class _FakeCompletions:
    def __init__(self, response=None, error=None) -> None:
        self.response = response or _chat_response()
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _provider_with_completions(completions: _FakeCompletions) -> OpenRouterTextProvider:
    return OpenRouterTextProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )


def test_openrouter_client_uses_openrouter_base_and_disables_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "generationengine.providers.openrouter_text.AsyncOpenAI",
        FakeAsyncOpenAI,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    OpenRouterTextProvider()
    assert captured["api_key"] == "or-test"
    assert captured["base_url"] == OPENROUTER_BASE_URL
    assert captured["max_retries"] == 0


def test_openrouter_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "generationengine.providers.openrouter_text.AsyncOpenAI",
        object,
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError) as exc:
        OpenRouterTextProvider()
    assert exc.value.failure.code is FailureCode.CONFIGURATION_UNAVAILABLE
    assert "OPENROUTER_API_KEY" in exc.value.failure.message


def test_request_id_is_http_id_not_completion_object_id() -> None:
    response = _chat_response()
    request_id, response_id = _ids_from_response(response)
    assert request_id == "req-or"
    assert response_id == "chatcmpl-or"


@pytest.mark.asyncio
async def test_openrouter_text_request_uses_chat_completions() -> None:
    completions = _FakeCompletions()
    provider = _provider_with_completions(completions)
    result = await provider.generate(
        TextGenerationCall(
            model="deepseek/deepseek-v4.1-flash",
            user_prompt="hello",
            system_prompt="sys",
            temperature=0.2,
        )
    )
    assert completions.calls[0]["model"] == "deepseek/deepseek-v4.1-flash"
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert completions.calls[0]["temperature"] == 0.2
    assert "input" not in completions.calls[0]
    assert result.text == "ok"
    assert result.provider_request_id == "req-or"
    assert result.provider_response_id == "chatcmpl-or"
    assert result.response_model == "deepseek/deepseek-v4.1-flash"
    assert result.input_tokens == 3
    assert result.cached_input_tokens == 1
    assert result.output_tokens == 2


@pytest.mark.asyncio
async def test_openrouter_text_request_omits_response_format() -> None:
    completions = _FakeCompletions()
    provider = _provider_with_completions(completions)
    await provider.generate(
        TextGenerationCall(
            model="deepseek/deepseek-v4.1-flash",
            user_prompt="hello",
        )
    )
    assert "response_format" not in completions.calls[0]
    assert "max_completion_tokens" not in completions.calls[0]


def test_openrouter_none_output_ceiling_omits_provider_field() -> None:
    provider = OpenRouterTextProvider(client=SimpleNamespace())
    kwargs = provider._request_kwargs(
        TextGenerationCall(
            model="deepseek/deepseek-v4.1-flash",
            user_prompt="hello",
            max_output_tokens=None,
        )
    )
    assert "max_completion_tokens" not in kwargs


def test_openrouter_output_ceiling_maps_to_completion_tokens() -> None:
    provider = OpenRouterTextProvider(client=SimpleNamespace())
    kwargs = provider._request_kwargs(
        TextGenerationCall(
            model="deepseek/deepseek-v4.1-flash",
            user_prompt="hello",
            max_output_tokens=400,
        )
    )
    assert kwargs["max_completion_tokens"] == 400


def test_openrouter_stream_output_ceiling_maps_to_completion_tokens() -> None:
    provider = OpenRouterTextProvider(client=SimpleNamespace())
    kwargs = provider._request_kwargs(
        TextGenerationCall(
            model="deepseek/deepseek-v4.1-flash",
            user_prompt="hello",
            max_output_tokens=400,
        ),
        streaming=True,
    )
    assert kwargs["max_completion_tokens"] == 400
    assert kwargs["stream"] is True


@pytest.mark.asyncio
async def test_openrouter_structured_uses_json_instruction_not_native_schema() -> None:
    completions = _FakeCompletions(response=_chat_response(text='{"name":"x","count":1}'))
    provider = _provider_with_completions(completions)
    result = await provider.generate(
        TextGenerationCall(
            model="deepseek/deepseek-v4.1-flash",
            user_prompt="hello",
            json_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["name", "count"],
            },
            schema_name="fixture",
        )
    )
    assert "response_format" not in completions.calls[0]
    system = completions.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert "JSON Schema" in system["content"]
    assert result.parsed is None
    assert result.text == '{"name":"x","count":1}'


@pytest.mark.asyncio
async def test_client_openrouter_generate_structured_uses_conformance() -> None:
    from generationengine import GenerationClient, TextRequest

    completions = _FakeCompletions(response=_chat_response(text='{"name":"ok","count":1}'))
    client = GenerationClient(
        text_providers={"openrouter": _provider_with_completions(completions)}
    )
    result = await client.generate_structured(
        TextRequest(
            user_prompt="hello",
            provider="openrouter",
            model="deepseek/deepseek-v4.1-flash",
            json_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["name", "count"],
                "additionalProperties": False,
            },
            max_output_tokens=400,
        )
    )
    assert result.parsed == {"name": "ok", "count": 1}
    assert result.observation.provider == "openrouter"
    assert result.observation.conformance_retry_count == 0
    assert completions.calls
    assert completions.calls[0]["max_completion_tokens"] == 400


@pytest.mark.asyncio
async def test_openrouter_maps_sdk_errors_to_safe_public_messages() -> None:
    provider = OpenRouterTextProvider(client=SimpleNamespace())
    error = provider._map_exception(
        RuntimeError("Authorization Bearer sk-or HTTP 502 from openrouter.ai")
    )
    assert error.failure.code is FailureCode.PROVIDER_ERROR
    assert error.failure.message == "Provider request failed."
    assert "sk-or" not in error.failure.message
    assert "openrouter.ai" not in error.failure.message

    class FakeTimeoutError(Exception):
        pass

    timeout = provider._map_exception(FakeTimeoutError("waited 45s"))
    assert timeout.failure.code is FailureCode.PROVIDER_TIMEOUT
    assert timeout.failure.message == "Provider request timed out."

    class FakeRateLimitError(Exception):
        pass

    rate_limited = provider._map_exception(
        FakeRateLimitError("429 https://openrouter.ai/api/v1/chat/completions Bearer sk-or")
    )
    assert rate_limited.failure.code is FailureCode.RATE_LIMITED
    assert rate_limited.failure.message == "Provider rate limit exceeded."
    assert "sk-or" not in rate_limited.failure.message


@pytest.mark.asyncio
async def test_openrouter_status_mapping() -> None:
    class StatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    provider = OpenRouterTextProvider(client=SimpleNamespace())
    assert provider._map_exception(StatusError(429)).failure.code is FailureCode.RATE_LIMITED
    assert provider._map_exception(StatusError(503)).failure.code is FailureCode.PROVIDER_UNAVAILABLE


@pytest.mark.asyncio
async def test_openrouter_stream_keeps_provider_identity() -> None:
    class StreamCompletions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True

            async def _chunks():
                yield SimpleNamespace(
                    id="chatcmpl-or",
                    _request_id="req-or",
                    model="deepseek/deepseek-v4.1-flash",
                    usage=None,
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason=None)],
                )
                yield SimpleNamespace(
                    id="chatcmpl-or",
                    model="deepseek/deepseek-v4.1-flash",
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        completion_tokens=1,
                        prompt_tokens_details=None,
                    ),
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")],
                )

            return _chunks()

    provider = OpenRouterTextProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=StreamCompletions()))
    )
    events = [
        event
        async for event in provider.stream(
            TextGenerationCall(model="deepseek/deepseek-v4.1-flash", user_prompt="hello")
        )
    ]
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "hi"
    assert isinstance(events[-1], TextCompleted)
    assert events[-1].final_text == "hi"
    assert events[-1].observation.provider == "openrouter"
    assert events[-1].observation.state is ObservationState.COMPLETED
    assert events[-1].observation.provider_request_id == "req-or"


@pytest.mark.asyncio
async def test_openrouter_stream_exception_is_failed_with_openrouter_identity() -> None:
    class Exploding:
        async def create(self, **kwargs):
            raise RuntimeError("socket died Bearer sk-or")

    provider = OpenRouterTextProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Exploding()))
    )
    events = [
        event
        async for event in provider.stream(
            TextGenerationCall(model="deepseek/deepseek-v4.1-flash", user_prompt="hello")
        )
    ]
    assert len(events) == 1
    assert isinstance(events[0], TextFailed)
    assert events[0].failure.code is FailureCode.PROVIDER_ERROR
    assert events[0].failure.message == "Provider request failed."
    assert events[0].observation.provider == "openrouter"
    assert "sk-or" not in events[0].failure.message


@pytest.mark.asyncio
async def test_openrouter_omitted_temperature_forwards_0_7() -> None:
    completions = _FakeCompletions()
    provider = _provider_with_completions(completions)
    await provider.generate(
        TextGenerationCall(model="deepseek/deepseek-v4.1-flash", user_prompt="hello")
    )
    assert completions.calls[0]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_openrouter_none_temperature_omits_provider_field() -> None:
    completions = _FakeCompletions()
    provider = _provider_with_completions(completions)
    await provider.generate(
        TextGenerationCall(
            model="deepseek/deepseek-v4.1-flash",
            user_prompt="hello",
            temperature=None,
        )
    )
    assert "temperature" not in completions.calls[0]


@pytest.mark.asyncio
async def test_openrouter_zero_temperature_is_forwarded() -> None:
    completions = _FakeCompletions()
    provider = _provider_with_completions(completions)
    await provider.generate(
        TextGenerationCall(
            model="deepseek/deepseek-v4.1-flash",
            user_prompt="hello",
            temperature=0.0,
        )
    )
    assert completions.calls[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_openrouter_stream_none_temperature_omits_provider_field() -> None:
    captured: dict[str, object] = {}

    class StreamCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)

            async def _chunks():
                yield SimpleNamespace(
                    id="chatcmpl-or",
                    _request_id="req-or",
                    model="deepseek/deepseek-v4.1-flash",
                    usage=None,
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason="stop")],
                )

            return _chunks()

    provider = OpenRouterTextProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=StreamCompletions()))
    )
    events = [
        event
        async for event in provider.stream(
            TextGenerationCall(
                model="deepseek/deepseek-v4.1-flash",
                user_prompt="hello",
                temperature=None,
            )
        )
    ]
    assert captured["stream"] is True
    assert "temperature" not in captured
    assert isinstance(events[-1], TextCompleted)


@pytest.mark.asyncio
async def test_openrouter_stream_numeric_temperature_is_forwarded() -> None:
    captured: dict[str, object] = {}

    class StreamCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)

            async def _chunks():
                yield SimpleNamespace(
                    id="chatcmpl-or",
                    _request_id="req-or",
                    model="deepseek/deepseek-v4.1-flash",
                    usage=None,
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason="stop")],
                )

            return _chunks()

    provider = OpenRouterTextProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=StreamCompletions()))
    )
    events = [
        event
        async for event in provider.stream(
            TextGenerationCall(
                model="deepseek/deepseek-v4.1-flash",
                user_prompt="hello",
                temperature=0.2,
            )
        )
    ]
    assert captured["temperature"] == 0.2
    assert isinstance(events[-1], TextCompleted)
