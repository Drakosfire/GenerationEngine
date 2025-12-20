"""Image generation response models."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from generationengine.models.metrics import GenerationMetrics
from generationengine.models.responses import GenerationError


class ImageResult(BaseModel):
    """Result for a single generated image."""

    url: str = Field(..., description="Cloudflare R2 URL of the uploaded image")
    width: int = Field(..., ge=1, description="Image width in pixels")
    height: int = Field(..., ge=1, description="Image height in pixels")
    model_used: str = Field(..., description="Model that generated this image")


class ImageGenerationResponse(BaseModel):
    """Response model for image generation."""

    success: bool = Field(..., description="Whether generation succeeded")
    images: Optional[list[ImageResult]] = Field(None, description="Generated images (present if success=True)")
    metrics: Optional[GenerationMetrics] = Field(None, description="Performance/cost tracking")
    error: Optional[GenerationError] = Field(None, description="Error details if success=False")

    @model_validator(mode="after")
    def validate_success_state(self):
        """Ensure success state is consistent."""
        if self.success is True:
            if not self.images:
                raise ValueError("images must be present when success=True")
            if self.error:
                raise ValueError("error must be None when success=True")
        else:
            if not self.error:
                raise ValueError("error must be present when success=False")
            if self.images:
                raise ValueError("images must be None when success=False")
        return self

