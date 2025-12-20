"""Response models for GenerationEngine."""

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator

from generationengine.models.errors import ErrorCode
from generationengine.models.metrics import GenerationMetrics

T = TypeVar("T")


class GenerationError(BaseModel):
    """Error details for failed generation operations."""

    code: ErrorCode = Field(..., description="Error category code")
    message: str = Field(..., description="User-friendly error message")
    retryable: bool = Field(..., description="Whether the client should retry this request")
    details: Optional[dict] = Field(None, description="Optional additional context for debugging")


class GenerationResponse(BaseModel, Generic[T]):
    """Standardized response wrapper for all generation operations."""

    success: bool = Field(..., description="Whether the generation operation succeeded")
    data: Optional[T] = Field(None, description="Generated content (type-specific)")
    metrics: Optional[GenerationMetrics] = Field(None, description="Performance/cost tracking")
    error: Optional[GenerationError] = Field(None, description="Error details if success=False")

    @model_validator(mode="after")
    def validate_success_state(self):
        """Ensure success state is consistent."""
        if self.success is True:
            # If success=True, data must be present, error must be None
            if self.data is None:
                raise ValueError("data must be present when success=True")
            if self.error is not None:
                raise ValueError("error must be None when success=True")
        else:
            # If success=False, error must be present, data must be None
            if self.error is None:
                raise ValueError("error must be present when success=False")
            if self.data is not None:
                raise ValueError("data must be None when success=False")
        return self

