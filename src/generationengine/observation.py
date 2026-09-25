"""Normalized inference-call observation.

This is inference-call truth, not a product trace. Unknown values are None.
Zero means the provider supplied zero. Full prompts and responses are not fields.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from generationengine.failures import FailureCode

FORBIDDEN_OBSERVATION_FIELDS = frozenset(
    {
        "prompt",
        "response",
        "raw_request",
        "raw_response",
        "input_payload",
        "output_payload",
        "system_prompt",
        "user_prompt",
    }
)


class ObservationState(str, Enum):
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class InferenceObservation(BaseModel):
    """One GenerationEngine inference operation.

    Structured generation may include multiple provider requests (transport
    retries and at most one conformance repair) under this same observation.
    """

    provider: str | None = None
    requested_profile: str | None = None
    requested_model: str | None = None
    resolved_model: str | None = None
    response_model: str | None = None
    provider_request_id: str | None = None
    provider_response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    latency_ms: int = Field(..., ge=0)
    retry_count: int = Field(..., ge=0, description="Transport retries: additional provider attempts after the first try of a provider call")
    transport_retry_count: int | None = Field(
        default=None,
        ge=0,
        description="Same meaning as retry_count. Filled from retry_count when omitted.",
    )
    conformance_retry_count: int = Field(
        default=0,
        ge=0,
        description="Corrective structured-generation attempts after the initial provider call succeeded with invalid structure",
    )
    provider_attempt_count: int | None = Field(
        default=None,
        ge=0,
        description="Total provider generate invocations across transport and conformance attempts",
    )
    state: ObservationState
    failure_code: FailureCode | None = None
    pricing_source: str | None = None

    @model_validator(mode="after")
    def validate_state_and_failure(self) -> InferenceObservation:
        if self.transport_retry_count is None:
            object.__setattr__(self, "transport_retry_count", self.retry_count)
        if self.state is ObservationState.COMPLETED:
            if self.failure_code is not None:
                raise ValueError("completed observations must not include failure_code")
        elif self.failure_code is None:
            raise ValueError(f"{self.state.value} observations require failure_code")
        return self
