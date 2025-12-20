"""Text generation response models."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from generationengine.models.metrics import GenerationMetrics
from generationengine.models.responses import GenerationError


class TextGenerationResponse(BaseModel):
    """Response model for text generation."""

    success: bool = Field(..., description="Whether generation succeeded")
    content: Optional[str] = Field(None, description="Generated text content (present if success=True)")
    metrics: Optional[GenerationMetrics] = Field(None, description="Performance/cost tracking")
    error: Optional[GenerationError] = Field(None, description="Error details if success=False")

    @model_validator(mode="after")
    def validate_success_state(self):
        """Ensure success state is consistent."""
        if self.success is True:
            if not self.content:
                raise ValueError("content must be present when success=True")
            if self.error:
                raise ValueError("error must be None when success=True")
        else:
            if not self.error:
                raise ValueError("error must be present when success=False")
            if self.content:
                raise ValueError("content must be None when success=False")
        return self

