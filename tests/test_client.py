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
