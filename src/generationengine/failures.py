"""Target failure taxonomy for GenerationEngine.

Current provider execution may still use ErrorCode / RetryableError during E2B.
This module is the contract the coordinated cutover should emit. It is not an
adapter that keeps both taxonomies alive.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FailureCode(str, Enum):
    CONFIGURATION_UNAVAILABLE = "CONFIGURATION_UNAVAILABLE"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    INVALID_REQUEST = "INVALID_REQUEST"
    PROVIDER_REFUSED = "PROVIDER_REFUSED"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    MALFORMED_PROVIDER_RESPONSE = "MALFORMED_PROVIDER_RESPONSE"
    STRUCTURED_OUTPUT_INVALID = "STRUCTURED_OUTPUT_INVALID"
    STREAM_INCOMPLETE = "STREAM_INCOMPLETE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class Retryability(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


FAILURE_RETRYABILITY: dict[FailureCode, Retryability] = {
    FailureCode.CONFIGURATION_UNAVAILABLE: Retryability.NO,
    FailureCode.UNSUPPORTED_CAPABILITY: Retryability.NO,
    FailureCode.INVALID_REQUEST: Retryability.NO,
    FailureCode.PROVIDER_REFUSED: Retryability.NO,
    FailureCode.RATE_LIMITED: Retryability.YES,
    FailureCode.PROVIDER_TIMEOUT: Retryability.YES,
    FailureCode.PROVIDER_UNAVAILABLE: Retryability.YES,
    FailureCode.PROVIDER_ERROR: Retryability.UNKNOWN,
    FailureCode.MALFORMED_PROVIDER_RESPONSE: Retryability.NO,
    FailureCode.STRUCTURED_OUTPUT_INVALID: Retryability.NO,
    FailureCode.STREAM_INCOMPLETE: Retryability.NO,
    FailureCode.INTERNAL_ERROR: Retryability.NO,
}


class InferenceFailure(BaseModel):
    """Public failure shape. Must not require provider SDK exception types."""

    code: FailureCode
    message: str = Field(..., min_length=1, description="Safe, non-secret message")
    retryability: Retryability

    @classmethod
    def from_code(cls, code: FailureCode, message: str) -> InferenceFailure:
        return cls(code=code, message=message, retryability=FAILURE_RETRYABILITY[code])
