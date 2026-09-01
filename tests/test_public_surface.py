"""Consumer-required GenerationEngine import paths and return shapes.

Pin demonstrated DungeonMindServer seams, not the historical
``generationengine.__all__`` snapshot. Unused baseline exports such as
``IGenerator`` may be retired during E2.
"""

from __future__ import annotations

import generationengine as ge
from generationengine.models.errors import ErrorCode
from generationengine.models.image_responses import ImageGenerationResponse, ImageResult
from generationengine.models.metrics import GenerationMetrics
from generationengine.models.responses import GenerationError
from generationengine.models.text_responses import TextGenerationResponse

# Package-root exports that DungeonMindServer imports today.
CONSUMER_ROOT_EXPORTS = (
    "ImageService",
    "MetricsService",
    "ImageGenerationRequest",
    "ImageModel",
    "ImageSize",
    "TextGenerationService",
    "TextGenerationRequest",
    "TextModel",
)

INTERNAL_CONSUMER_IMPORTS = (
    "generationengine.services.text_service.TextGenerationService",
    "generationengine.models.requests.TextGenerationRequest",
    "generationengine.models.requests.TextModel",
    "generationengine.models.image_responses.ImageGenerationResponse",
    "generationengine.models.image_responses.ImageResult",
)


def test_consumer_root_exports_remain_public() -> None:
    missing = [name for name in CONSUMER_ROOT_EXPORTS if name not in ge.__all__]
    assert missing == [], f"consumer exports missing from __all__: {missing}"
    for name in CONSUMER_ROOT_EXPORTS:
        assert hasattr(ge, name), name


def test_internal_consumer_import_paths_remain() -> None:
    for dotted in INTERNAL_CONSUMER_IMPORTS:
        module_name, attr = dotted.rsplit(".", 1)
        module = __import__(module_name, fromlist=[attr])
        assert hasattr(module, attr), dotted


def test_unused_baseline_exports_are_not_consumer_frozen() -> None:
    """E2 may retire these; they are inventory, not DungeonMindServer seams."""
    assert "IGenerator" not in CONSUMER_ROOT_EXPORTS
    assert "GenerationResponse" not in CONSUMER_ROOT_EXPORTS
    assert "RetryableError" not in CONSUMER_ROOT_EXPORTS
    assert "is_retryable" not in CONSUMER_ROOT_EXPORTS


def test_text_response_consumer_shape() -> None:
    ok = TextGenerationResponse(
        success=True,
        content="hello",
        parsed_content={"name": "x"},
        metrics=GenerationMetrics(duration_ms=12, tokens_used=30, model_used="gpt-5.1", retry_count=0),
    )
    assert ok.success is True
    assert ok.content == "hello"
    assert ok.parsed_content == {"name": "x"}
    assert ok.metrics is not None
    assert ok.metrics.tokens_used == 30

    failed = TextGenerationResponse(
        success=False,
        error=GenerationError(
            code=ErrorCode.INTERNAL_ERROR,
            message="generation failed",
            retryable=False,
        ),
    )
    assert failed.success is False
    assert failed.error is not None
    assert failed.error.message == "generation failed"


def test_image_response_consumer_shape() -> None:
    image = ImageResult(
        url="https://example.invalid/generated.png",
        width=1024,
        height=1024,
        model_used="gpt-image-1.5",
    )
    response = ImageGenerationResponse(
        success=True,
        images=[image],
        metrics=GenerationMetrics(
            duration_ms=40,
            model_used="gpt-image-1.5",
            retry_count=1,
        ),
    )
    assert response.success is True
    assert response.images is not None
    assert response.images[0].url == "https://example.invalid/generated.png"
    assert response.images[0].width == 1024
    assert response.images[0].height == 1024
    assert response.metrics is not None
    assert response.metrics.duration_ms == 40
    assert response.metrics.model_used == "gpt-image-1.5"
    assert response.metrics.retry_count == 1
