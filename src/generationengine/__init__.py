"""DungeonMind GenerationEngine - provider-agnostic inference execution."""

from __future__ import annotations

import importlib
from typing import Any

from generationengine.catalog import (
    ACCEPTED_CAPABILITIES,
    ACCEPTED_PROFILES,
    Availability,
    Capability,
    InferenceProfile,
    ModelRecord,
    PricingDimension,
)
from generationengine.failures import FailureCode, InferenceFailure, Retryability
from generationengine.models.errors import ErrorCode
from generationengine.models.metrics import GenerationMetrics
from generationengine.models.requests import (
    ImageGenerationRequest,
    ImageModel,
    ImageSize,
    TextGenerationRequest,
    TextModel,
)
from generationengine.models.responses import GenerationError
from generationengine.models.text_responses import TextGenerationResponse
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
from generationengine.services.metrics_service import MetricsService

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ImageService": ("generationengine.services.image_service", "ImageService"),
    "TextGenerationService": ("generationengine.services.text_service", "TextGenerationService"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACCEPTED_CAPABILITIES",
    "ACCEPTED_PROFILES",
    "Availability",
    "Capability",
    "ErrorCode",
    "FailureCode",
    "GenerationError",
    "GenerationMetrics",
    "ImageGenerationRequest",
    "ImageModel",
    "ImageProvider",
    "ImageService",
    "ImageSize",
    "InferenceFailure",
    "InferenceObservation",
    "InferenceProfile",
    "MetricsService",
    "ModelRecord",
    "ObservationState",
    "PricingDimension",
    "Retryability",
    "TextCompleted",
    "TextDelta",
    "TextFailed",
    "TextGenerationCall",
    "TextGenerationRequest",
    "TextGenerationResponse",
    "TextGenerationResult",
    "TextGenerationService",
    "TextModel",
    "TextProvider",
    "TextStreamEvent",
]
