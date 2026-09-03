"""Response models for GenerationEngine."""

from typing import Optional

from pydantic import BaseModel, Field

from generationengine.models.errors import ErrorCode


class GenerationError(BaseModel):
    """Error details for failed generation operations."""

    code: ErrorCode = Field(..., description="Error category code")
    message: str = Field(..., description="User-friendly error message")
    retryable: bool = Field(..., description="Whether the client should retry this request")
    details: Optional[dict] = Field(None, description="Optional additional context for debugging")
