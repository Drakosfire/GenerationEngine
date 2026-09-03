"""DungeonMind GenerationEngine - provider-agnostic inference execution."""

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
    TextDelta,
    TextGenerationCall,
    TextGenerationResult,
    TextProvider,
)
from generationengine.services.image_service import ImageService
from generationengine.services.metrics_service import MetricsService
from generationengine.services.text_service import TextGenerationService

__version__ = "0.1.0"

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
    "TextDelta",
    "TextGenerationCall",
    "TextGenerationRequest",
    "TextGenerationResponse",
    "TextGenerationResult",
    "TextGenerationService",
    "TextModel",
    "TextProvider",
]
