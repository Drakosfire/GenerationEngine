"""Provider protocols for GenerationEngine.

These seams are for the coordinated cutover. E2B does not move live OpenAI/Fal
execution behind them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import runtime_checkable

from generationengine.failures import InferenceFailure
from generationengine.observation import InferenceObservation


class TextGenerationCall(BaseModel):
    """Provider-neutral text call. No product schemas or SSE framing."""

    model: str
    user_prompt: str
    system_prompt: str | None = None
    temperature: float = 0.7
    json_schema: dict[str, Any] | None = None
    schema_name: str | None = None


class TextDelta(BaseModel):
    text: str


class TextCompleted(BaseModel):
    final_text: str
    observation: InferenceObservation


class TextFailed(BaseModel):
    failure: InferenceFailure
    observation: InferenceObservation


TextStreamEvent = Union[TextDelta, TextCompleted, TextFailed]


class TextGenerationResult(BaseModel):
    text: str | None = None
    parsed: dict[str, Any] | None = None
    refused: bool = False
    provider_request_id: str | None = None
    response_model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


@runtime_checkable
class TextProvider(Protocol):
    """Execute text / structured text without exposing SDK types."""

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        """Return a completed text result."""
        ...

    def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
        """Yield transport-neutral stream events ending in TextCompleted or TextFailed."""
        ...


@runtime_checkable
class ImageProvider(Protocol):
    """Protocol for image generation providers."""

    async def generate(
        self,
        prompt: str,
        model: str,
        num_images: int,
        size: Tuple[int, int],
        image_url: str | None = None,
        strength: float | None = None,
        mask_base64: str | None = None,
        base_image_base64: str | None = None,
        negative_prompt: str | None = None,
    ) -> list[bytes]:
        """
        Generate images from a prompt.

        Returns image bytes. Durable publication is not part of this protocol.
        """
        ...
