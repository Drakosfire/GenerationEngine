"""Retry service with exponential backoff for generation operations."""

import asyncio
from typing import Any, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from generationengine.models.errors import ErrorCode, is_retryable

T = TypeVar("T")

# Standard retry configuration: 3 retries at 1s, 2s, 4s intervals
DEFAULT_RETRY_CONFIG = {
    "stop": stop_after_attempt(3),  # 1 initial attempt + 2 retries = 3 total
    "wait": wait_exponential(multiplier=1, min=1, max=4),  # 1s, 2s, 4s
    "reraise": True,
}


class RetryableError(Exception):
    """Base exception for retryable errors."""

    def __init__(self, error_code: ErrorCode, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.original_exception = original_exception


async def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    retry_config: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with retry logic and exponential backoff.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        retry_config: Optional custom retry configuration. If None, uses default.
        timeout_seconds: Optional timeout per attempt. If None, no timeout is applied.
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        RetryableError: If all retries are exhausted or timeout occurs
        Exception: Non-retryable exceptions are re-raised immediately
    """
    config = retry_config or DEFAULT_RETRY_CONFIG.copy()
    config.setdefault("retry", retry_if_exception_type(RetryableError))

    async def _execute_with_timeout():
        """Execute func with optional timeout."""
        if timeout_seconds is not None:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError as e:
                raise RetryableError(
                    ErrorCode.PROVIDER_TIMEOUT,
                    f"Request timed out after {timeout_seconds}s",
                    original_exception=e,
                )
        else:
            return await func(*args, **kwargs)

    async for attempt in AsyncRetrying(**config):
        with attempt:
            return await _execute_with_timeout()


def should_retry(error_code: ErrorCode) -> bool:
    """Check if an error code indicates a retryable error."""
    return is_retryable(error_code)

