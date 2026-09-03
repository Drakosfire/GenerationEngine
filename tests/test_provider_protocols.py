"""Provider protocol seams exist without product or SSE vocabulary."""

from __future__ import annotations

from collections.abc import AsyncIterator

from generationengine.providers.base import (
    ImageProvider,
    TextDelta,
    TextGenerationCall,
    TextGenerationResult,
    TextProvider,
)


class _FakeText:
    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        return TextGenerationResult(text=call.user_prompt, input_tokens=None, output_tokens=0)

    async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextDelta]:
        yield TextDelta(text=call.user_prompt)


class _FakeImage:
    async def generate(self, prompt, model, num_images, size, **kwargs) -> list[bytes]:
        return [b"x"] * num_images


def test_text_and_image_provider_protocols_are_structural() -> None:
    assert isinstance(_FakeText(), TextProvider)
    assert isinstance(_FakeImage(), ImageProvider)
    call = TextGenerationCall(model="gpt-5.1", user_prompt="hello")
    assert "data:" not in call.model_dump_json()
    assert "statblock" not in TextGenerationCall.model_json_schema().__repr__().lower()
