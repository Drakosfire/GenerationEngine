"""Public execution request/result types. No provider SDK types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from generationengine.catalog import InferenceProfile
from generationengine.failures import InferenceFailure
from generationengine.observation import InferenceObservation


class GenerationEngineError(Exception):
    """Public failure. Carries normalized failure + observation; never an SDK type."""

    def __init__(self, failure: InferenceFailure, observation: InferenceObservation) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.observation = observation


class TextRequest(BaseModel):
    user_prompt: str
    system_prompt: str | None = None
    profile: InferenceProfile | None = None
    model: str | None = None
    temperature: float = 0.7
    json_schema: dict[str, Any] | None = None
    schema_name: str | None = None
    deadline_ms: int | None = Field(default=None, ge=1)


class TextResult(BaseModel):
    text: str | None = None
    parsed: dict[str, Any] | None = None
    observation: InferenceObservation


class ImageRequest(BaseModel):
    prompt: str
    profile: InferenceProfile | None = None
    model: str | None = None
    num_images: int = Field(default=1, ge=1, le=8)
    width: int = Field(default=1024, ge=1)
    height: int = Field(default=1024, ge=1)
    negative_prompt: str | None = None
    source_image_url: str | None = None
    source_image_bytes: bytes | None = None
    mask_base64: str | None = None
    base_image_base64: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    deadline_ms: int | None = Field(default=None, ge=1)


class GeneratedImage(BaseModel):
    content: bytes
    media_type: str = "image/png"
    width: int | None = None
    height: int | None = None


class ImageResult(BaseModel):
    images: list[GeneratedImage]
    observation: InferenceObservation
