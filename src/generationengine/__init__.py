"""DungeonMind GenerationEngine - provider-agnostic inference execution."""

from __future__ import annotations

from generationengine.catalog import (
    ACCEPTED_CAPABILITIES,
    ACCEPTED_PROFILES,
    Availability,
    Capability,
    InferenceProfile,
    ModelRecord,
    PricingDimension,
)
from generationengine.client import GenerationClient
from generationengine.failures import FailureCode, InferenceFailure, Retryability
from generationengine.observation import InferenceObservation, ObservationState
from generationengine.providers.base import (
    DecisionGenerationCall,
    DecisionGenerationResult,
    DecisionProvider,
    ImageProvider,
    TextCompleted,
    TextDelta,
    TextFailed,
    TextGenerationCall,
    TextGenerationResult,
    TextProvider,
    TextStreamEvent,
)
from generationengine.resolver import LIVE_MODELS, ResolutionError, resolve
from generationengine.types import (
    BinaryDecisionAnswer,
    BinaryDecisionQuestion,
    ChoiceDecisionAnswer,
    ChoiceDecisionQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    GeneratedImage,
    GenerationEngineError,
    ImageRequest,
    ImageResult,
    ScoreDecisionAnswer,
    ScoreDecisionQuestion,
    TextRequest,
    TextResult,
)

__version__ = "0.1.0"

__all__ = [
    "BinaryDecisionAnswer",
    "BinaryDecisionQuestion",
    "ACCEPTED_CAPABILITIES",
    "ACCEPTED_PROFILES",
    "Availability",
    "Capability",
    "ChoiceDecisionAnswer",
    "ChoiceDecisionQuestion",
    "DecisionAnswer",
    "DecisionGenerationCall",
    "DecisionGenerationResult",
    "DecisionProvider",
    "DecisionQuestion",
    "DecisionRequest",
    "DecisionResult",
    "FailureCode",
    "GeneratedImage",
    "GenerationClient",
    "GenerationEngineError",
    "ImageProvider",
    "ImageRequest",
    "ImageResult",
    "InferenceFailure",
    "InferenceObservation",
    "InferenceProfile",
    "LIVE_MODELS",
    "ModelRecord",
    "ObservationState",
    "PricingDimension",
    "ResolutionError",
    "Retryability",
    "ScoreDecisionAnswer",
    "ScoreDecisionQuestion",
    "TextCompleted",
    "TextDelta",
    "TextFailed",
    "TextGenerationCall",
    "TextGenerationResult",
    "TextProvider",
    "TextRequest",
    "TextResult",
    "TextStreamEvent",
    "resolve",
]
