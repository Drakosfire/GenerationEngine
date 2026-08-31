"""Compatibility inventory for current public and consumer import paths.

These names are KEEP TEMPORARILY / COMPATIBILITY surfaces. Adding target types
is allowed. Removing a name from this list requires an E3 consumer migration,
not an E2A cleanup.
"""

from __future__ import annotations

import generationengine as ge

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

# Current __all__ snapshot at E2A baseline. May grow; must not silently shrink
# without a compatibility note in docs/COMPATIBILITY.md.
BASELINE_PUBLIC_EXPORTS = (
    "IGenerator",
    "GenerationResponse",
    "GenerationError",
    "GenerationMetrics",
    "ErrorCode",
    "is_retryable",
    "ImageGenerationRequest",
    "ImageModel",
    "ImageSize",
    "TextGenerationRequest",
    "TextGenerationResponse",
    "TextModel",
    "ImageProvider",
    "ImageService",
    "MetricsService",
    "TextGenerationService",
    "make_schema_strict",
    "RetryableError",
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


def test_baseline_public_exports_remain_importable() -> None:
    missing = [name for name in BASELINE_PUBLIC_EXPORTS if name not in ge.__all__]
    assert missing == [], f"baseline public exports removed: {missing}"


def test_internal_consumer_import_paths_remain() -> None:
    for dotted in INTERNAL_CONSUMER_IMPORTS:
        module_name, attr = dotted.rsplit(".", 1)
        module = __import__(module_name, fromlist=[attr])
        assert hasattr(module, attr), dotted
