"""Contract tests for GenerationResponse error shape and validation."""

import pytest

from generationengine.models.errors import ErrorCode
from generationengine.models.responses import GenerationError, GenerationResponse


def test_generation_error_shape():
    """T057: Contract test for GenerationResponse error shape."""
    error = GenerationError(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="Request timed out",
        retryable=True,
    )
    
    # Verify error has required fields
    assert hasattr(error, "code")
    assert hasattr(error, "message")
    assert hasattr(error, "retryable")
    
    # Verify types
    assert isinstance(error.code, ErrorCode)
    assert isinstance(error.message, str)
    assert isinstance(error.retryable, bool)
    
    # Verify values
    assert error.code == ErrorCode.PROVIDER_TIMEOUT
    assert error.message == "Request timed out"
    assert error.retryable is True


def test_generation_response_error_shape():
    """T057: Contract test for GenerationResponse error shape (full response)."""
    error = GenerationError(
        code=ErrorCode.RATE_LIMITED,
        message="Rate limit exceeded",
        retryable=True,
    )
    
    response = GenerationResponse[str](
        success=False,
        error=error,
    )
    
    # Verify response structure
    assert response.success is False
    assert response.error is not None
    assert response.data is None
    
    # Verify error shape matches contract
    assert response.error.code == ErrorCode.RATE_LIMITED
    assert response.error.message == "Rate limit exceeded"
    assert response.error.retryable is True


def test_generation_response_success_shape():
    """Contract test for GenerationResponse success shape."""
    response = GenerationResponse[str](
        success=True,
        data="Generated content",
    )
    
    assert response.success is True
    assert response.data == "Generated content"
    assert response.error is None


def test_generation_response_validation_fails_on_inconsistent_success():
    """Test that GenerationResponse validation fails when success state is inconsistent."""
    error = GenerationError(
        code=ErrorCode.INVALID_INPUT,
        message="Invalid input",
        retryable=False,
    )
    
    # success=True but error is present
    with pytest.raises(ValueError, match="error must be None when success=True"):
        GenerationResponse[str](
            success=True,
            data="content",
            error=error,
        )


def test_generation_response_validation_fails_on_inconsistent_failure():
    """Test that GenerationResponse validation fails when failure state is inconsistent."""
    # success=False but no error
    with pytest.raises(ValueError, match="error must be present when success=False"):
        GenerationResponse[str](
            success=False,
        )

