"""Provider protocol seams exist without product or SSE vocabulary."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from generationengine.failures import FailureCode, InferenceFailure
from generationengine.observation import InferenceObservation, ObservationState
from generationengine.providers.base import (
    ImageProvider,
    TextCompleted,
    TextDelta,
    TextFailed,
    TextGenerationCall,
    TextGenerationResult,
    TextProvider,
    TextStreamEvent,
)


class _FakeText:
    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        return TextGenerationResult(text=call.user_prompt, input_tokens=None, output_tokens=0)

    async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
        yield TextDelta(text="partial ")
        yield TextCompleted(
            final_text=call.user_prompt,
            observation=InferenceObservation(
                latency_ms=1,
                retry_count=0,
                state=ObservationState.COMPLETED,
            ),
        )


class _FakeImage:
    async def generate(self, prompt, model, num_images, size, **kwargs) -> list[bytes]:
        return [b"x"] * num_images


def test_text_and_image_provider_protocols_are_structural() -> None:
    assert isinstance(_FakeText(), TextProvider)
    assert isinstance(_FakeImage(), ImageProvider)
    call = TextGenerationCall(model="gpt-5.1", user_prompt="hello")
    assert "data:" not in call.model_dump_json()
    assert "statblock" not in TextGenerationCall.model_json_schema().__repr__().lower()


@pytest.mark.asyncio
async def test_text_provider_stream_emits_terminal_completion() -> None:
    events: list[TextStreamEvent] = []
    async for event in _FakeText().stream(TextGenerationCall(model="gpt-5.1", user_prompt="done")):
        events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], TextDelta)
    assert isinstance(events[1], TextCompleted)
    assert events[1].final_text == "done"
    assert events[1].observation.state == ObservationState.COMPLETED


@pytest.mark.asyncio
async def test_text_provider_stream_failure_event_shape() -> None:
    class _FailingText:
        async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
            raise NotImplementedError

        async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
            observation = InferenceObservation(
                latency_ms=5,
                retry_count=0,
                state=ObservationState.FAILED,
                failure_code=FailureCode.PROVIDER_UNAVAILABLE.value,
            )
            yield TextFailed(
                failure=InferenceFailure.from_code(
                    FailureCode.PROVIDER_UNAVAILABLE,
                    "provider down",
                ),
                observation=observation,
            )

    events = [event async for event in _FailingText().stream(TextGenerationCall(model="x", user_prompt="y"))]
    assert len(events) == 1
    assert isinstance(events[0], TextFailed)
    assert events[0].failure.code == FailureCode.PROVIDER_UNAVAILABLE
