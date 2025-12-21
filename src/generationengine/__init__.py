"""DungeonMind GenerationEngine - Unified generation infrastructure."""

from generationengine.interfaces import IGenerator
from generationengine.models.errors import ErrorCode, is_retryable
from generationengine.models.metrics import GenerationMetrics
from generationengine.models.requests import (
    ImageGenerationRequest,
    ImageModel,
    ImageSize,
    TextGenerationRequest,
    TextModel,
)
from generationengine.models.responses import GenerationError, GenerationResponse
from generationengine.models.text_responses import TextGenerationResponse
from generationengine.providers.base import ImageProvider
from generationengine.services.image_service import ImageService
from generationengine.services.metrics_service import MetricsService
from generationengine.services.retry_service import RetryableError
from generationengine.services.text_service import TextGenerationService
from generationengine.utils.schema_utils import make_schema_strict

__version__ = "0.1.0"

__all__ = [
    # Interfaces
    "IGenerator",
    # Response/Error types
    "GenerationResponse",
    "GenerationError",
    "GenerationMetrics",
    "ErrorCode",
    "is_retryable",
    # Request types
    "ImageGenerationRequest",
    "ImageModel",
    "ImageSize",
    "TextGenerationRequest",
    "TextGenerationResponse",
    "TextModel",
    # Providers
    "ImageProvider",
    # Services
    "ImageService",
    "MetricsService",
    "TextGenerationService",
    # Utilities
    "make_schema_strict",
    "RetryableError",
]

