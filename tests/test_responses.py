"""Contract tests for GenerationError. Unused GenerationResponse was removed in E2B."""

from generationengine.models.errors import ErrorCode
from generationengine.models.responses import GenerationError


def test_generation_error_shape():
    error = GenerationError(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="Request timed out",
        retryable=True,
    )

    assert error.code == ErrorCode.PROVIDER_TIMEOUT
    assert error.message == "Request timed out"
    assert error.retryable is True
