"""Request models for GenerationEngine."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ImageModel(str, Enum):
    """AI models available for image generation via Fal.ai."""

    # Primary generation models (text-to-image and inpainting)
    FLUX_2_PRO = "flux-2-pro"  # fal-ai/flux-2-pro, fal-ai/flux-2-pro/edit
    NANO_BANANA_PRO = "nano-banana-pro"  # fal-ai/nano-banana-pro, fal-ai/nano-banana-pro/edit
    GPT_IMAGE_15 = "gpt-image-1.5"  # fal-ai/gpt-image-1.5, fal-ai/gpt-image-1.5/edit
    FLUX_PRO = "flux-pro"  # fal-ai/flux-pro (text-to-image only)
    FLUX_LORA_I2I = "flux-lora-i2i"  # fal-ai/flux-lora/image-to-image


class ImageSize(str, Enum):
    """Output image dimensions."""

    SQUARE = "square"  # 1024x1024
    PORTRAIT = "portrait"  # 768x1024
    LANDSCAPE = "landscape"  # 1024x768


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="Image generation prompt")
    negative_prompt: Optional[str] = Field(
        None,
        max_length=1000,
        description="Elements to exclude from the image (e.g., 'grid, text, characters'). Supported by most models."
    )
    model: ImageModel = Field(ImageModel.FLUX_2_PRO, description="Model to use for generation")
    num_images: int = Field(4, ge=1, le=8, description="Number of images to generate (1-8)")
    size: ImageSize = Field(ImageSize.SQUARE, description="Output image dimensions")
    image_url: Optional[str] = Field(
        None,
        description="Source image URL for image-to-image generation. When provided, the model transforms this image instead of generating from scratch."
    )
    strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Transformation strength for image-to-image (0.0-1.0). Higher values create more dramatic changes. Defaults to 0.85 if image_url is provided."
    )
    mask_base64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG mask for inpainting. Transparent areas (alpha=0) will be generated, opaque areas (alpha=1) will be preserved. Must be provided with base_image_base64."
    )
    base_image_base64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG base image for inpainting. Content will be generated within masked regions. Must be provided with mask_base64."
    )

    def get_size_tuple(self) -> tuple[int, int]:
        """Convert ImageSize enum to (width, height) tuple."""
        size_map: dict[ImageSize, tuple[int, int]] = {
            ImageSize.SQUARE: (1024, 1024),
            ImageSize.PORTRAIT: (768, 1024),
            ImageSize.LANDSCAPE: (1024, 768),
        }
        return size_map[self.size]


class TextModel(str, Enum):
    """AI models available for text generation."""

    GPT_5_1 = "gpt-5.1"


class TextGenerationRequest(BaseModel):
    """Request model for text generation."""

    system_prompt: str | None = Field(None, description="System message for the AI")
    user_prompt: str = Field(..., min_length=1, description="User message/prompt")
    model: TextModel = Field(TextModel.GPT_5_1, description="Model to use for generation")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate (None = model default)")
    response_schema: dict[str, Any] | None = Field(
        None,
        description="JSON schema for structured output. When provided, OpenAI's structured outputs feature is used to guarantee the response matches this schema."
    )
    response_schema_name: str = Field(
        "structured_response",
        description="Name for the response schema (used in OpenAI API)"
    )

