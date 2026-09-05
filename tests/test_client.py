"""Public execution client: resolution, retries, observations, stream terminals."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from generationengine import (
    Capability,
    FailureCode,
    GenerationClient,
    GenerationEngineError,
    InferenceProfile,
    ObservationState,
    TextCompleted,
    TextDelta,
    TextFailed,
    TextGenerationCall,
    TextGenerationResult,
    TextRequest,
    TextStreamEvent,
)
from generationengine.observation import InferenceObservation
from generationengine.providers.errors import ProviderError
from generationengine.resolver import resolve


class FakeTextProvider:
    def __init__(self, *, results=None, errors=None, stream_events=None) -> None:
        self.results = list(results or [])
        self.errors = list(errors or [])
        self.stream_events = stream_events
        self.calls = 0

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.results:
            return self.results.pop(0)
        return TextGenerationResult(
            text="ok",
            parsed={"name": "x", "count": 1} if call.json_schema else None,
            provider_request_id="req-1",
            response_model=call.model,
            input_tokens=10,
            cached_input_tokens=0,
            output_tokens=4,
        )

    async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
        if self.stream_events is not None:
            for event in self.stream_events:
                yield event
            return
        yield TextDelta(text="hel")
        yield TextDelta(text="lo")
        result = await self.generate(call)
        yield TextCompleted(
            final_text=result.text or "",
            observation=InferenceObservation(
                provider="openai",
                resolved_model=call.model,
                latency_ms=1,
                retry_count=0,
                state=ObservationState.COMPLETED,
            ),
        )


class FakeImageProvider:
    def __init__(self, blobs: list[bytes] | None = None) -> None:
        self.blobs = blobs or [b"png-bytes"]
        self.calls = 0

    async def generate(self, **kwargs) -> list[bytes]:
        self.calls += 1
        return list(self.blobs)


def test_profile_resolution_preserves_cutover_models() -> None:
    assert resolve(capability=Capability.TEXT, profile=InferenceProfile.TEXT_FAST).catalog_id == "gpt-5.1"
    assert (
        resolve(
            capability=Capability.STRUCTURED_TEXT,
            profile=InferenceProfile.STRUCTURED_HIGH_RELIABILITY,
        ).catalog_id
        == "gpt-5.6-luna"
    )
    assert (
        resolve(capability=Capability.IMAGE, model="nano-banana-pro").catalog_id
        == "nano-banana-pro"
    )
    explicit = resolve(
        capability=Capability.STRUCTURED_TEXT,
        profile=InferenceProfile.STRUCTURED_LOW_COST,
        model="gpt-4o",
    )
    assert explicit.catalog_id == "gpt-4o"


def test_unsupported_model_capability() -> None:
    with pytest.raises(Exception) as exc:
        resolve(capability=Capability.IMAGE, model="gpt-5.1")
    assert exc.value.failure.code is FailureCode.UNSUPPORTED_CAPABILITY


@pytest.mark.asyncio
async def test_text_success_and_observation() -> None:
    client = GenerationClient(text_provider=FakeTextProvider())
    result = await client.generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
    )
    assert result.text == "ok"
    assert result.observation.state is ObservationState.COMPLETED
    assert result.observation.resolved_model == "gpt-5.1"
    assert result.observation.requested_profile == "text_fast"
    assert result.observation.provider_request_id == "req-1"
    assert result.observation.input_tokens == 10
    assert result.observation.cached_input_tokens == 0
    assert result.observation.output_tokens == 4
    assert result.observation.cost_usd is None
    assert result.observation.retry_count == 0
    assert result.observation.failure_code is None


@pytest.mark.asyncio
async def test_structured_success() -> None:
    client = GenerationClient(text_provider=FakeTextProvider())
    result = await client.generate_structured(
        TextRequest(
            user_prompt="hi",
            profile=InferenceProfile.STRUCTURED_LOW_COST,
            json_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["name", "count"],
            },
            schema_name="fixture",
        )
    )
    assert result.parsed == {"name": "x", "count": 1}


@pytest.mark.asyncio
async def test_rate_limit_retries_then_succeeds() -> None:
    provider = FakeTextProvider(
        errors=[ProviderError.from_code(FailureCode.RATE_LIMITED, "slow")],
        results=[TextGenerationResult(text="recovered", provider_request_id="req-2")],
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
    )
    assert result.text == "recovered"
    assert result.observation.retry_count == 1
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_timeout_maps_to_provider_timeout() -> None:
    provider = FakeTextProvider(
        errors=[
            ProviderError.from_code(FailureCode.PROVIDER_TIMEOUT, "too slow"),
            ProviderError.from_code(FailureCode.PROVIDER_TIMEOUT, "too slow"),
            ProviderError.from_code(FailureCode.PROVIDER_TIMEOUT, "too slow"),
        ]
    )
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_text(TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST))
    assert exc.value.failure.code is FailureCode.PROVIDER_TIMEOUT
    assert exc.value.observation.retry_count == 2
    assert exc.value.observation.state is ObservationState.FAILED


@pytest.mark.asyncio
async def test_refusal_is_not_success() -> None:
    provider = FakeTextProvider(
        results=[TextGenerationResult(text=None, refused=True, provider_request_id="r")],
    )
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_text(TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST))
    assert exc.value.failure.code is FailureCode.PROVIDER_REFUSED
    assert exc.value.observation.state is ObservationState.REFUSED


@pytest.mark.asyncio
async def test_stream_terminals_exactly_once() -> None:
    client = GenerationClient(text_provider=FakeTextProvider())
    events = [
        event
        async for event in client.stream_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        )
    ]
    terminals = [event for event in events if isinstance(event, (TextCompleted, TextFailed))]
    assert len(terminals) == 1
    assert isinstance(terminals[0], TextCompleted)
    assert any(isinstance(event, TextDelta) for event in events)


@pytest.mark.asyncio
async def test_stream_incomplete_without_terminal() -> None:
    provider = FakeTextProvider(stream_events=[TextDelta(text="partial")])
    client = GenerationClient(text_provider=provider)
    events = [
        event
        async for event in client.stream_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        )
    ]
    assert isinstance(events[-1], TextFailed)
    assert events[-1].failure.code is FailureCode.STREAM_INCOMPLETE


@pytest.mark.asyncio
async def test_image_returns_bytes_not_url() -> None:
    client = GenerationClient(image_provider=FakeImageProvider([b"img-a", b"img-b"]))
    from generationengine.types import ImageRequest

    result = await client.generate_image(
        ImageRequest(prompt="a map", model="gpt-image-1.5", num_images=2)
    )
    assert [image.content for image in result.images] == [b"img-a", b"img-b"]
    assert result.observation.state is ObservationState.COMPLETED
    assert result.observation.resolved_model == "gpt-image-1.5"
    assert all(not hasattr(image, "url") for image in result.images)


@pytest.mark.asyncio
async def test_missing_profile_or_model() -> None:
    client = GenerationClient(text_provider=FakeTextProvider())
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_text(TextRequest(user_prompt="hi"))
    assert exc.value.failure.code is FailureCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_deadline_is_overall_budget_not_per_attempt(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _no_sleep(_delay: float) -> None:
        sleeps.append(_delay)

    monkeypatch.setattr("generationengine.client.asyncio.sleep", _no_sleep)
    provider = FakeTextProvider(
        errors=[
            ProviderError.from_code(FailureCode.RATE_LIMITED, "slow"),
            ProviderError.from_code(FailureCode.RATE_LIMITED, "slow"),
            ProviderError.from_code(FailureCode.RATE_LIMITED, "slow"),
        ]
    )
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST, deadline_ms=80)
        )
    assert exc.value.failure.code is FailureCode.RATE_LIMITED
    assert exc.value.failure.message == "Provider rate limit exceeded."
    assert provider.calls == 1
    assert exc.value.observation.retry_count == 0
    assert sleeps == []


@pytest.mark.asyncio
async def test_retries_use_backoff_when_budget_allows(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("generationengine.client.asyncio.sleep", _record_sleep)
    provider = FakeTextProvider(
        errors=[
            ProviderError.from_code(FailureCode.RATE_LIMITED, "slow"),
            ProviderError.from_code(FailureCode.RATE_LIMITED, "slow"),
        ],
        results=[TextGenerationResult(text="recovered")],
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST, deadline_ms=10_000)
    )
    assert result.text == "recovered"
    assert result.observation.retry_count == 2
    assert provider.calls == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_image_wait_for_timeout_is_provider_timeout() -> None:
    import asyncio

    class SlowImage:
        async def generate(self, **kwargs) -> list[bytes]:
            await asyncio.sleep(1)
            return [b"late"]

    client = GenerationClient(image_provider=SlowImage())
    from generationengine.types import ImageRequest

    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_image(
            ImageRequest(prompt="a map", model="gpt-image-1.5", deadline_ms=50)
        )
    assert exc.value.failure.code is FailureCode.PROVIDER_TIMEOUT
    assert exc.value.failure.message == "Provider request timed out."
    assert exc.value.observation.state is ObservationState.FAILED


@pytest.mark.asyncio
async def test_stream_observation_is_client_owned() -> None:
    import asyncio

    class SlowStream(FakeTextProvider):
        async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
            await asyncio.sleep(0.02)
            yield TextCompleted(
                final_text="hello",
                observation=InferenceObservation(
                    provider="openai",
                    provider_request_id="http-req",
                    provider_response_id="resp_abc",
                    latency_ms=0,
                    retry_count=0,
                    state=ObservationState.COMPLETED,
                ),
            )
            yield TextCompleted(
                final_text="second-terminal",
                observation=InferenceObservation(
                    provider="openai",
                    latency_ms=0,
                    retry_count=0,
                    state=ObservationState.COMPLETED,
                ),
            )

    client = GenerationClient(text_provider=SlowStream())
    events = [
        event
        async for event in client.stream_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        )
    ]
    terminals = [event for event in events if isinstance(event, (TextCompleted, TextFailed))]
    assert len(terminals) == 1
    completed = terminals[0]
    assert isinstance(completed, TextCompleted)
    assert completed.final_text == "hello"
    assert completed.observation.requested_profile == "text_fast"
    assert completed.observation.resolved_model == "gpt-5.1"
    assert completed.observation.provider == "openai"
    assert completed.observation.provider_request_id == "http-req"
    assert completed.observation.provider_response_id == "resp_abc"
    assert completed.observation.latency_ms >= 20
    assert completed.observation.retry_count == 0


@pytest.mark.asyncio
async def test_stream_partial_then_deadline_is_timeout() -> None:
    import asyncio

    class PartialThenHang(FakeTextProvider):
        async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
            yield TextDelta(text="partial")
            await asyncio.sleep(1)
            yield TextCompleted(
                final_text="too-late",
                observation=InferenceObservation(
                    provider="openai",
                    latency_ms=0,
                    retry_count=0,
                    state=ObservationState.COMPLETED,
                ),
            )

    client = GenerationClient(text_provider=PartialThenHang())
    events = [
        event
        async for event in client.stream_text(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.TEXT_FAST,
                deadline_ms=50,
            )
        )
    ]
    terminals = [event for event in events if isinstance(event, (TextCompleted, TextFailed))]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["partial"]
    assert len(terminals) == 1
    assert isinstance(terminals[0], TextFailed)
    assert terminals[0].failure.code is FailureCode.PROVIDER_TIMEOUT
    assert terminals[0].failure.message == "Provider request timed out."
    assert terminals[0].observation.retry_count == 0
    assert terminals[0].observation.requested_profile == "text_fast"


@pytest.mark.asyncio
async def test_stream_config_failure_before_delta_is_terminal(monkeypatch) -> None:
    def _boom(_self):
        raise ProviderError.from_code(
            FailureCode.CONFIGURATION_UNAVAILABLE,
            "OPENAI_API_KEY is required for text generation.",
        )

    monkeypatch.setattr(
        "generationengine.providers.openai_text.OpenAITextProvider.__init__",
        _boom,
    )
    client = GenerationClient()
    events = [
        event
        async for event in client.stream_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        )
    ]
    assert len(events) == 1
    assert isinstance(events[0], TextFailed)
    assert events[0].failure.code is FailureCode.CONFIGURATION_UNAVAILABLE
    assert events[0].observation.state is ObservationState.FAILED


@pytest.mark.asyncio
async def test_stream_provider_exception_during_stream_is_terminal() -> None:
    class ExplodingStream(FakeTextProvider):
        async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
            yield TextDelta(text="partial")
            raise RuntimeError("socket died")

    client = GenerationClient(text_provider=ExplodingStream())
    events = [
        event
        async for event in client.stream_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        )
    ]
    terminals = [event for event in events if isinstance(event, (TextCompleted, TextFailed))]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["partial"]
    assert len(terminals) == 1
    assert isinstance(terminals[0], TextFailed)
    assert terminals[0].failure.code is FailureCode.PROVIDER_ERROR
    assert terminals[0].failure.message == "Provider request failed."
    assert "socket died" not in terminals[0].failure.message


@pytest.mark.asyncio
async def test_stream_duplicate_provider_terminal_emits_one() -> None:
    failed = TextFailed(
        failure=ProviderError.from_code(FailureCode.PROVIDER_ERROR, "first").failure,
        observation=InferenceObservation(
            provider="openai",
            latency_ms=0,
            retry_count=0,
            state=ObservationState.FAILED,
            failure_code=FailureCode.PROVIDER_ERROR,
        ),
    )
    duplicate = TextFailed(
        failure=ProviderError.from_code(FailureCode.PROVIDER_ERROR, "second").failure,
        observation=InferenceObservation(
            provider="openai",
            latency_ms=0,
            retry_count=0,
            state=ObservationState.FAILED,
            failure_code=FailureCode.PROVIDER_ERROR,
        ),
    )
    provider = FakeTextProvider(stream_events=[TextDelta(text="partial"), failed, duplicate])
    client = GenerationClient(text_provider=provider)
    events = [
        event
        async for event in client.stream_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        )
    ]
    terminals = [event for event in events if isinstance(event, (TextCompleted, TextFailed))]
    assert len(terminals) == 1
    assert isinstance(terminals[0], TextFailed)
    assert terminals[0].failure.code is FailureCode.PROVIDER_ERROR
    assert terminals[0].failure.message == "Provider request failed."


def test_map_provider_exception_does_not_leak_exception_text() -> None:
    from generationengine.client import _map_provider_exception

    leaked = RuntimeError("Authorization Bearer sk-live https://api.openai.com/v1/responses")
    error = _map_provider_exception(leaked)
    assert error.failure.code is FailureCode.PROVIDER_ERROR
    assert error.failure.message == "Provider request failed."
    assert "sk-live" not in error.failure.message
    assert "openai.com" not in error.failure.message

    timeout = _map_provider_exception(TimeoutError("waited 45s for sk-live"))
    assert timeout.failure.code is FailureCode.PROVIDER_TIMEOUT
    assert timeout.failure.message == "Provider request timed out."
    assert "sk-live" not in timeout.failure.message
