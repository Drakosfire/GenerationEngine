"""Error code definitions for GenerationEngine."""

from enum import Enum


class ErrorCode(str, Enum):
    """Error category codes for generation operations."""

    # Retryable errors (retryable=True)
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_OVERLOADED = "PROVIDER_OVERLOADED"
    RATE_LIMITED = "RATE_LIMITED"

    # Not retryable errors (retryable=False)
    INVALID_INPUT = "INVALID_INPUT"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    NOT_FOUND = "NOT_FOUND"
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Set of retryable error codes
RETRYABLE_ERRORS = {
    ErrorCode.PROVIDER_TIMEOUT,
    ErrorCode.PROVIDER_OVERLOADED,
    ErrorCode.RATE_LIMITED,
}


def is_retryable(code: ErrorCode) -> bool:
    """Check if an error code indicates a retryable error."""
    return code in RETRYABLE_ERRORS

