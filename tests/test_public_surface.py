"""Public package surface after the E2 cutover."""

from __future__ import annotations

import generationengine as ge

PUBLIC = (
    "GenerationClient",
    "GenerationEngineError",
    "TextRequest",
    "TextResult",
    "ImageRequest",
    "ImageResult",
    "GeneratedImage",
    "InferenceObservation",
    "FailureCode",
    "InferenceFailure",
    "InferenceProfile",
    "TextProvider",
    "ImageProvider",
    "TextCompleted",
    "TextFailed",
    "TextStreamEvent",
)


def test_new_contract_is_exported() -> None:
    missing = [name for name in PUBLIC if name not in ge.__all__]
    assert missing == []
    for name in PUBLIC:
        assert hasattr(ge, name)


def test_legacy_facades_are_gone() -> None:
    for name in (
        "TextGenerationService",
        "ImageService",
        "TextGenerationRequest",
        "TextModel",
        "ImageGenerationRequest",
        "ImageModel",
        "ImageSize",
        "MetricsService",
        "GenerationMetrics",
        "IGenerator",
        "GenerationResponse",
    ):
        assert name not in ge.__all__
        assert not hasattr(ge, name)
