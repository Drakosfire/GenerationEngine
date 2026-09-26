"""Public execution client: resolution, retries, observations, stream terminals."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from generationengine import (
    Capability,
    FailureCode,
    GenerationClient,
    GenerationEngineError,
    ImageRequest,
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
from generationengine.client import _map_provider_exception
from generationengine.observation import InferenceObservation
from generationengine.providers.errors import ProviderError
from generationengine.resolver import resolve


class FakeTextProvider:
    def __init__(self, *, results=None, errors=None, stream_events=None) -> None:
        self.results = list(results or [])
        self.errors = list(errors or [])
        self.stream_events = stream_events
        self.calls = 0
        self.seen_calls: list[TextGenerationCall] = []

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        self.calls += 1
        self.seen_calls.append(call)
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


class CloseableProvider(FakeTextProvider):
    def __init__(self, *, close_error: Exception | None = None) -> None:
        super().__init__()
        self.close_error = close_error
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def test_aclose_is_async() -> None:
    assert inspect.iscoroutinefunction(GenerationClient.aclose)


@pytest.mark.asyncio
async def test_aclose_unused_lazy_client_constructs_no_provider() -> None:
    client = GenerationClient.from_env()

    await client.aclose()

    assert client._text_providers == {}
    assert client._image is None


@pytest.mark.asyncio
async def test_aclose_closes_unique_provider_once_and_is_idempotent() -> None:
    provider = CloseableProvider()
    client = GenerationClient(
        text_provider=provider,
        text_providers={"openrouter": provider},
        image_provider=provider,
    )

    await client.aclose()
    await client.aclose()

    assert provider.close_calls == 1


@pytest.mark.asyncio
async def test_aclose_skips_provider_without_async_close() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)

    await client.aclose()

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_aclose_attempts_all_providers_and_reraises_first_error() -> None:
    first_error = RuntimeError("first close failed")
    first = CloseableProvider(close_error=first_error)
    second = CloseableProvider(close_error=ValueError("second close failed"))
    last = CloseableProvider()
    client = GenerationClient(
        text_providers={"openai": first, "openrouter": second},
        image_provider=last,
    )

    with pytest.raises(RuntimeError) as exc:
        await client.aclose()

    assert exc.value is first_error
    assert [first.close_calls, second.close_calls, last.close_calls] == [1, 1, 1]
    await client.aclose()
    assert [first.close_calls, second.close_calls, last.close_calls] == [1, 1, 1]
    with pytest.raises(GenerationEngineError) as inference_exc:
        await client.generate_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        )
    assert inference_exc.value.failure.code is FailureCode.INVALID_REQUEST
    assert inference_exc.value.observation.provider_attempt_count == 0
    assert first.calls == 0


@pytest.mark.asyncio
async def test_closed_client_rejects_text_structured_and_images_before_provider() -> None:
    text = FakeTextProvider()
    image = FakeImageProvider()
    client = GenerationClient(text_provider=text, image_provider=image)
    await client.aclose()

    operations = (
        client.generate_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        ),
        client.generate_structured(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.STRUCTURED_LOW_COST,
                json_schema={"type": "object"},
            )
        ),
        client.generate_image(ImageRequest(prompt="map", model="gpt-image-1.5")),
        client.edit_image(
            ImageRequest(
                prompt="map",
                model="gpt-image-1.5",
                base_image_base64="aW1hZ2U=",
            )
        ),
    )
    for operation in operations:
        with pytest.raises(GenerationEngineError) as exc:
            await operation
        assert exc.value.failure.code is FailureCode.INVALID_REQUEST
        assert exc.value.observation.provider_attempt_count == 0

    assert text.calls == 0
    assert image.calls == 0


@pytest.mark.asyncio
async def test_closed_client_stream_yields_one_invalid_terminal_without_provider_call() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)
    await client.aclose()

    events = [
        event
        async for event in client.stream_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], TextFailed)
    assert events[0].failure.code is FailureCode.INVALID_REQUEST
    assert events[0].observation.provider_attempt_count == 0
    assert provider.calls == 0


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


@pytest.mark.parametrize("value", [0, -1])
def test_text_output_token_ceiling_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError):
        TextRequest(user_prompt="hi", max_output_tokens=value)
    with pytest.raises(ValueError):
        TextGenerationCall(model="gpt-5.1", user_prompt="hi", max_output_tokens=value)


def test_text_output_token_ceiling_defaults_to_none() -> None:
    assert TextRequest(user_prompt="hi").max_output_tokens is None
    assert TextGenerationCall(model="gpt-5.1", user_prompt="hi").max_output_tokens is None


def test_json_object_mode_defaults_to_false() -> None:
    assert TextRequest(user_prompt="hi").json_object is False
    assert TextGenerationCall(model="gpt-5.1", user_prompt="hi").json_object is False


@pytest.mark.asyncio
async def test_json_object_mode_reaches_provider_and_returns_raw_text() -> None:
    provider = FakeTextProvider(
        results=[
            TextGenerationResult(
                text='{"name":"raw"}',
                parsed={"provider": "must not cross the JSON-object boundary"},
            )
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_text(
        TextRequest(
            user_prompt="hi",
            profile=InferenceProfile.TEXT_FAST,
            json_object=True,
        )
    )
    assert provider.seen_calls[0].json_object is True
    assert result.text == '{"name":"raw"}'
    assert result.parsed is None


@pytest.mark.asyncio
async def test_json_object_mode_returns_malformed_json_without_repair() -> None:
    provider = FakeTextProvider(
        results=[TextGenerationResult(text="{not-json", parsed=None)]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_text(
        TextRequest(
            user_prompt="hi",
            profile=InferenceProfile.TEXT_FAST,
            json_object=True,
        )
    )
    assert result.text == "{not-json"
    assert result.parsed is None
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_json_object_mode_transport_retry_preserves_request() -> None:
    provider = FakeTextProvider(
        errors=[ProviderError.from_code(FailureCode.RATE_LIMITED, "slow")],
        results=[TextGenerationResult(text="{}", parsed=None)],
    )
    client = GenerationClient(text_provider=provider)
    await client.generate_text(
        TextRequest(
            user_prompt="hi",
            profile=InferenceProfile.TEXT_FAST,
            json_object=True,
        )
    )
    assert [call.json_object for call in provider.seen_calls] == [True, True]


@pytest.mark.asyncio
async def test_json_object_and_schema_reject_before_provider_execution() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_text(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.TEXT_FAST,
                json_object=True,
                json_schema={"type": "object"},
            )
        )
    assert exc.value.failure.code is FailureCode.INVALID_REQUEST
    assert exc.value.observation.provider_attempt_count == 0
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_generate_structured_rejects_json_object_before_provider_execution() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_structured(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.STRUCTURED_LOW_COST,
                json_object=True,
                json_schema={"type": "object"},
            )
        )
    assert exc.value.failure.code is FailureCode.INVALID_REQUEST
    assert exc.value.observation.provider_attempt_count == 0
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_stream_rejects_json_object_before_provider_execution() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)
    events = [
        event
        async for event in client.stream_text(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.TEXT_FAST,
                json_object=True,
            )
        )
    ]
    assert len(events) == 1
    assert isinstance(events[0], TextFailed)
    assert events[0].failure.code is FailureCode.INVALID_REQUEST
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_text_output_token_ceiling_reaches_provider_unchanged() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)
    await client.generate_text(
        TextRequest(
            user_prompt="hi",
            profile=InferenceProfile.TEXT_FAST,
            max_output_tokens=400,
        )
    )
    assert [call.max_output_tokens for call in provider.seen_calls] == [400]


@pytest.mark.asyncio
async def test_omitted_output_token_ceiling_reaches_provider_as_none() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)
    await client.generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
    )
    assert [call.max_output_tokens for call in provider.seen_calls] == [None]


@pytest.mark.asyncio
async def test_transport_retry_preserves_output_token_ceiling() -> None:
    provider = FakeTextProvider(
        errors=[ProviderError.from_code(FailureCode.RATE_LIMITED, "slow")],
        results=[TextGenerationResult(text="recovered")],
    )
    client = GenerationClient(text_provider=provider)
    await client.generate_text(
        TextRequest(
            user_prompt="hi",
            profile=InferenceProfile.TEXT_FAST,
            max_output_tokens=400,
        )
    )
    assert [call.max_output_tokens for call in provider.seen_calls] == [400, 400]


@pytest.mark.asyncio
async def test_structured_repair_preserves_output_token_ceiling() -> None:
    class RepairProvider(FakeTextProvider):
        async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
            self.calls += 1
            self.seen_calls.append(call)
            if self.calls == 1:
                return TextGenerationResult(text="not-json", parsed=None)
            return TextGenerationResult(text='{"name":"ok"}', parsed=None)

    provider = RepairProvider()
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(
        TextRequest(
            user_prompt="hi",
            profile=InferenceProfile.STRUCTURED_LOW_COST,
            json_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            max_output_tokens=400,
        )
    )
    assert result.parsed == {"name": "ok"}
    assert [call.max_output_tokens for call in provider.seen_calls] == [400, 400]


@pytest.mark.asyncio
async def test_stream_preserves_output_token_ceiling() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)
    events = [
        event
        async for event in client.stream_text(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.TEXT_FAST,
                max_output_tokens=400,
            )
        )
    ]
    assert isinstance(events[-1], TextCompleted)
    assert [call.max_output_tokens for call in provider.seen_calls] == [400]


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
async def test_default_retry_terminal_failure_reports_all_provider_attempts() -> None:
    provider = FakeTextProvider(
        errors=[ProviderError.from_code(FailureCode.RATE_LIMITED) for _ in range(3)]
    )
    client = GenerationClient(text_provider=provider)

    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST, deadline_ms=10_000)
        )

    assert provider.calls == 3
    assert exc.value.observation.retry_count == 2
    assert exc.value.observation.provider_attempt_count == 3


@pytest.mark.asyncio
async def test_incomplete_text_failure_is_nonretryable_and_preserves_observation() -> None:
    provider = FakeTextProvider(
        errors=[ProviderError.from_code(
            FailureCode.PROVIDER_INCOMPLETE,
            "SECRET partial output",
            provider_request_id="req-incomplete",
            provider_response_id="resp-incomplete",
            response_model="gpt-5.1",
            input_tokens=8,
            cached_input_tokens=0,
            output_tokens=2,
            reasoning_tokens=1,
        )]
    )
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient(text_provider=provider).generate_text(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.TEXT_FAST,
                max_transport_retries=3,
            )
        )
    assert provider.calls == 1
    assert exc.value.failure.code is FailureCode.PROVIDER_INCOMPLETE
    assert exc.value.failure.message == "Provider returned an incomplete response."
    assert exc.value.observation.state is ObservationState.INCOMPLETE
    assert exc.value.observation.provider_attempt_count == 1
    assert exc.value.observation.transport_retry_count == 0
    assert exc.value.observation.conformance_retry_count == 0
    assert exc.value.observation.provider_request_id == "req-incomplete"
    assert exc.value.observation.provider_response_id == "resp-incomplete"
    assert exc.value.observation.response_model == "gpt-5.1"
    assert (exc.value.observation.input_tokens, exc.value.observation.cached_input_tokens, exc.value.observation.output_tokens, exc.value.observation.reasoning_tokens) == (8, 0, 2, 1)
    assert "SECRET" not in str(exc.value.observation.model_dump())


def test_provider_error_rebuild_preserves_reasoning_usage() -> None:
    original = ProviderError.from_code(
        FailureCode.PROVIDER_INCOMPLETE,
        reasoning_tokens=0,
    )
    rebuilt = _map_provider_exception(original)
    assert rebuilt.reasoning_tokens == 0
    assert rebuilt.failure.code is FailureCode.PROVIDER_INCOMPLETE


@pytest.mark.asyncio
async def test_explicit_transport_retry_ceiling_and_reasoning_pass_through() -> None:
    for ceiling, expected_calls in ((0, 1), (1, 2), (3, 4)):
        provider = FakeTextProvider(
            errors=[ProviderError.from_code(FailureCode.RATE_LIMITED) for _ in range(expected_calls)]
        )
        client = GenerationClient(text_provider=provider)
        with pytest.raises(GenerationEngineError) as exc:
            await client.generate_text(
                TextRequest(
                    user_prompt="hi",
                    profile=InferenceProfile.TEXT_FAST,
                    reasoning_effort="high",
                    max_transport_retries=ceiling,
                    deadline_ms=10_000,
                )
            )
        assert provider.calls == expected_calls
        assert exc.value.observation.retry_count == ceiling
        assert exc.value.observation.provider_attempt_count == expected_calls
        assert all(call.reasoning_effort == "high" for call in provider.seen_calls)
        assert all(call.max_transport_retries == ceiling for call in provider.seen_calls)

    assert TextRequest(user_prompt="hi").max_transport_retries is None
    assert TextRequest(user_prompt="hi").reasoning_effort is None
    with pytest.raises(ValidationError):
        TextRequest(user_prompt="hi", max_transport_retries=-1)
    with pytest.raises(ValidationError):
        TextRequest(user_prompt="hi", reasoning_effort="")


@pytest.mark.asyncio
async def test_nonretryable_error_and_deadline_bound_requested_retries() -> None:
    nonretryable = FakeTextProvider(
        errors=[ProviderError.from_code(FailureCode.INVALID_REQUEST, "Invalid provider request.")]
    )
    with pytest.raises(GenerationEngineError) as nonretryable_exc:
        await GenerationClient(text_provider=nonretryable).generate_text(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.TEXT_FAST,
                max_transport_retries=3,
            )
        )
    assert nonretryable.calls == 1
    assert nonretryable_exc.value.observation.provider_attempt_count == 1

    class SlowProvider(FakeTextProvider):
        async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
            self.calls += 1
            await asyncio.sleep(0.02)
            return TextGenerationResult(text="late")

    slow = SlowProvider()
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient(text_provider=slow).generate_text(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.TEXT_FAST,
                max_transport_retries=3,
                deadline_ms=1,
            )
        )
    assert exc.value.failure.code is FailureCode.PROVIDER_TIMEOUT
    # A short deadline may expire before the coroutine enters the provider.
    assert slow.calls <= 1
    assert exc.value.observation.provider_attempt_count == slow.calls


@pytest.mark.asyncio
async def test_stream_positive_retry_rejected_before_provider_and_zero_keeps_stream() -> None:
    provider = FakeTextProvider()
    client = GenerationClient(text_provider=provider)
    events = [
        event async for event in client.stream_text(
            TextRequest(user_prompt="hi", max_transport_retries=1)
        )
    ]
    assert len(events) == 1
    assert isinstance(events[0], TextFailed)
    assert events[0].failure.code is FailureCode.INVALID_REQUEST
    assert events[0].observation.provider_attempt_count == 0
    assert provider.calls == 0
    assert client._text_providers == {"openai": provider}

    lazy_client = GenerationClient.from_env()
    lazy_events = [
        event async for event in lazy_client.stream_text(
            TextRequest(user_prompt="hi", max_transport_retries=1)
        )
    ]
    assert len(lazy_events) == 1
    assert isinstance(lazy_events[0], TextFailed)
    assert lazy_client._text_providers == {}

    events = [
        event async for event in client.stream_text(
            TextRequest(
                user_prompt="hi",
                profile=InferenceProfile.TEXT_FAST,
                reasoning_effort="low",
                max_transport_retries=0,
            )
        )
    ]
    assert isinstance(events[-1], TextCompleted)
    assert provider.seen_calls[0].reasoning_effort == "low"


@pytest.mark.asyncio
async def test_ordinary_and_stream_reasoning_usage_is_observed() -> None:
    provider = FakeTextProvider(
        results=[TextGenerationResult(text="ok", reasoning_tokens=0)]
    )
    result = await GenerationClient(text_provider=provider).generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
    )
    assert result.observation.reasoning_tokens == 0

    class StreamUsageProvider(FakeTextProvider):
        async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
            yield TextCompleted(
                final_text="ok",
                observation=InferenceObservation(
                    provider="openai",
                    reasoning_tokens=7,
                    latency_ms=0,
                    retry_count=0,
                    state=ObservationState.COMPLETED,
                ),
            )

    events = [
        event async for event in GenerationClient(
            text_provider=StreamUsageProvider()
        ).stream_text(TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST))
    ]
    assert isinstance(events[-1], TextCompleted)
    assert events[-1].observation.reasoning_tokens == 7


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


class RecordingTextProvider(FakeTextProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.received: list[TextGenerationCall] = []

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        self.received.append(call)
        return await super().generate(call)


def test_omitted_temperature_defaults_to_0_7() -> None:
    request = TextRequest(user_prompt="x", profile=InferenceProfile.TEXT_FAST)
    assert request.temperature == 0.7
    call = TextGenerationCall(model="gpt-5.1", user_prompt="x")
    assert call.temperature == 0.7


def test_explicit_none_temperature_is_preserved_on_request() -> None:
    request = TextRequest(
        user_prompt="x",
        profile=InferenceProfile.TEXT_FAST,
        temperature=None,
    )
    assert request.temperature is None
    call = TextGenerationCall(model="gpt-5.1", user_prompt="x", temperature=None)
    assert call.temperature is None


@pytest.mark.asyncio
async def test_omitted_temperature_reaches_provider_as_0_7() -> None:
    provider = RecordingTextProvider()
    client = GenerationClient(text_provider=provider)
    await client.generate_text(TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST))
    assert provider.received[0].temperature == 0.7


@pytest.mark.asyncio
async def test_explicit_numeric_temperature_reaches_provider() -> None:
    provider = RecordingTextProvider()
    client = GenerationClient(text_provider=provider)
    await client.generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST, temperature=0.2)
    )
    assert provider.received[0].temperature == 0.2


@pytest.mark.asyncio
async def test_explicit_none_temperature_reaches_provider_call() -> None:
    provider = RecordingTextProvider()
    client = GenerationClient(text_provider=provider)
    await client.generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST, temperature=None)
    )
    assert provider.received[0].temperature is None


@pytest.mark.asyncio
async def test_zero_temperature_is_numeric_not_missing() -> None:
    provider = RecordingTextProvider()
    client = GenerationClient(text_provider=provider)
    await client.generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST, temperature=0.0)
    )
    assert provider.received[0].temperature == 0.0


@pytest.mark.asyncio
async def test_stream_explicit_none_temperature_reaches_provider_call() -> None:
    provider = RecordingTextProvider()
    client = GenerationClient(text_provider=provider)
    events = [
        event
        async for event in client.stream_text(
            TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST, temperature=None)
        )
    ]
    assert provider.received[0].temperature is None
    assert any(isinstance(event, TextCompleted) for event in events)


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
