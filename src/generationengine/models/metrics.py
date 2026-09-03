"""Metrics models for GenerationEngine."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GenerationMetrics(BaseModel):
    """Tracking data for a generation operation."""

    model_config = ConfigDict()

    duration_ms: int = Field(..., ge=0, description="Total generation time in milliseconds")
    tokens_used: Optional[int] = Field(None, ge=0, description="Token count for text generation")
    estimated_cost_usd: Optional[float] = Field(None, ge=0.0, description="Estimated cost in USD")
    model_used: Optional[str] = Field(None, description="AI model identifier")
    retry_count: int = Field(
        0, ge=0, description="Number of retries performed (0 = first attempt succeeded)"
    )
    timestamp: Optional[datetime] = Field(None, description="When the generation completed (UTC)")
    input: Optional[str] = Field(None, description="Bounded input metadata as JSON string")
    output: Optional[str] = Field(
        None, description="Output summary as JSON string (counts/lengths only)"
    )
