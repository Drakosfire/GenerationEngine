"""Request models for GenerationEngine."""

from enum import Enum

from pydantic import BaseModel, Field


class ImageModel(str, Enum):
    """AI models available for image generation."""

    FLUX_PRO = "flux-pro"
    IMAGEN4 = "imagen4"
    IMAGEN4_FAST = "imagen4-fast"
    OPENAI = "openai"


class ImageSize(str, Enum):
    """Output image dimensions."""

    SQUARE = "square"  # 1024x1024
    PORTRAIT = "portrait"  # 768x1024
    LANDSCAPE = "landscape"  # 1024x768


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""

    prompt: str = Field(..., min_length=1, max_length=2000, description="Image generation prompt")
    model: ImageModel = Field(ImageModel.FLUX_PRO, description="Model to use for generation")
    num_images: int = Field(4, ge=1, le=8, description="Number of images to generate (1-8)")
    size: ImageSize = Field(ImageSize.SQUARE, description="Output image dimensions")

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

    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class TextGenerationRequest(BaseModel):
    """Request model for text generation."""

    system_prompt: str | None = Field(None, description="System message for the AI")
    user_prompt: str = Field(..., min_length=1, description="User message/prompt")
    model: TextModel = Field(TextModel.GPT_4O, description="Model to use for generation")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate (None = model default)")

