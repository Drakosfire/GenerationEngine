"""Tests for image generation service and models."""

import pytest
from datetime import datetime

from generationengine.models.requests import ImageGenerationRequest, ImageModel, ImageSize
from generationengine.models.image_responses import ImageGenerationResponse, ImageResult
from generationengine.models.errors import ErrorCode
from generationengine.models.metrics import GenerationMetrics


# Contract tests for ImageGenerationRequest
def test_image_generation_request_validation_valid():
    """Test that valid ImageGenerationRequest passes validation."""
    request = ImageGenerationRequest(
        prompt="A red dragon",
        model=ImageModel.FLUX_PRO,
        num_images=4,
        size=ImageSize.SQUARE,
    )

    assert request.prompt == "A red dragon"
    assert request.model == ImageModel.FLUX_PRO
    assert request.num_images == 4
    assert request.size == ImageSize.SQUARE


def test_image_generation_request_defaults():
    """Test that ImageGenerationRequest uses correct defaults."""
    request = ImageGenerationRequest(prompt="A red dragon")

    assert request.model == ImageModel.FLUX_PRO
    assert request.num_images == 4
    assert request.size == ImageSize.SQUARE


def test_image_generation_request_prompt_required():
    """Test that prompt is required."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImageGenerationRequest()


def test_image_generation_request_num_images_range():
    """Test that num_images is constrained to 1-8."""
    # Valid ranges
    ImageGenerationRequest(prompt="test", num_images=1)
    ImageGenerationRequest(prompt="test", num_images=8)

    # Invalid ranges
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImageGenerationRequest(prompt="test", num_images=0)

    with pytest.raises(Exception):  # Pydantic ValidationError
        ImageGenerationRequest(prompt="test", num_images=9)


def test_image_generation_request_get_size_tuple():
    """Test that get_size_tuple returns correct dimensions."""
    request_square = ImageGenerationRequest(prompt="test", size=ImageSize.SQUARE)
    assert request_square.get_size_tuple() == (1024, 1024)

    request_portrait = ImageGenerationRequest(prompt="test", size=ImageSize.PORTRAIT)
    assert request_portrait.get_size_tuple() == (768, 1024)

    request_landscape = ImageGenerationRequest(prompt="test", size=ImageSize.LANDSCAPE)
    assert request_landscape.get_size_tuple() == (1024, 768)


def test_image_generation_request_image_to_image_fields():
    """Test that ImageGenerationRequest accepts image-to-image parameters."""
    request = ImageGenerationRequest(
        prompt="Transform this image",
        model=ImageModel.FAL_FLUX_LORA_I2I,
        image_url="https://example.com/template.png",
        strength=0.85,
    )

    assert request.image_url == "https://example.com/template.png"
    assert request.strength == 0.85
    assert request.model == ImageModel.FAL_FLUX_LORA_I2I


def test_image_generation_request_strength_validation():
    """Test that strength is constrained to 0.0-1.0."""
    # Valid range
    ImageGenerationRequest(prompt="test", strength=0.0)
    ImageGenerationRequest(prompt="test", strength=1.0)
    ImageGenerationRequest(prompt="test", strength=0.5)

    # Invalid ranges
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImageGenerationRequest(prompt="test", strength=-0.1)

    with pytest.raises(Exception):  # Pydantic ValidationError
        ImageGenerationRequest(prompt="test", strength=1.1)


# Contract tests for ImageGenerationResponse
def test_image_generation_response_success_valid():
    """Test that successful ImageGenerationResponse has required fields."""
    response = ImageGenerationResponse(
        success=True,
        images=[
            ImageResult(
                url="https://r2.example.com/img1.png",
                width=1024,
                height=1024,
                model_used="flux-pro",
            )
        ],
        metrics=GenerationMetrics(duration_ms=1500),
    )

    assert response.success is True
    assert response.images is not None
    assert len(response.images) == 1
    assert response.error is None


def test_image_generation_response_failure_valid():
    """Test that failed ImageGenerationResponse has required fields."""
    from generationengine.models.responses import GenerationError

    response = ImageGenerationResponse(
        success=False,
        error=GenerationError(
            code=ErrorCode.PROVIDER_TIMEOUT,
            message="Generation timed out",
            retryable=True,
        ),
    )

    assert response.success is False
    assert response.error is not None
    assert response.images is None


def test_image_generation_response_success_requires_images():
    """Test that successful response must have images."""

    with pytest.raises(ValueError, match="images must be present when success=True"):
        ImageGenerationResponse(
            success=True,
            images=None,
        )


def test_image_generation_response_failure_requires_error():
    """Test that failed response must have error."""
    with pytest.raises(ValueError, match="error must be present when success=False"):
        ImageGenerationResponse(
            success=False,
            error=None,
        )


def test_image_generation_response_shape_matches_contract():
    """Test that response shape matches GenerationResponse contract (success, data/error, metrics)."""
    # Success case
    response = ImageGenerationResponse(
        success=True,
        images=[
            ImageResult(
                url="https://r2.example.com/img1.png",
                width=1024,
                height=1024,
                model_used="flux-pro",
            )
        ],
        metrics=GenerationMetrics(
            duration_ms=1500,
            model_used="flux-pro",
            timestamp=datetime.now(),
        ),
    )

    # Verify contract compliance
    assert hasattr(response, "success")
    assert hasattr(response, "images")  # Equivalent to "data" in generic contract
    assert hasattr(response, "metrics")
    assert hasattr(response, "error")
    assert isinstance(response.success, bool)
    assert response.metrics is not None

