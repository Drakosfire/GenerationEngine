"""Shared OpenAI-compatible SDK helpers.

Provider identity stays with the calling adapter. This module only maps
OpenAI-compatible transport exceptions and SDK import checks.
"""

from __future__ import annotations

from generationengine.failures import FailureCode
from generationengine.providers.errors import ProviderError

try:
    from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
except ImportError:
    AsyncOpenAI = None  # type: ignore
    RateLimitError = None  # type: ignore
    APITimeoutError = None  # type: ignore
    APIError = None  # type: ignore


def require_openai_sdk(*, extra_name: str, client_cls: object | None = None) -> None:
    resolved = AsyncOpenAI if client_cls is None else client_cls
    if resolved is None:
        raise ProviderError.from_code(
            FailureCode.CONFIGURATION_UNAVAILABLE,
            f"openai extra is not installed. Install generationengine[{extra_name}].",
        )


def is_sdk_exception(exc: Exception, sdk_type: type[Exception] | None) -> bool:
    return sdk_type is not None and isinstance(exc, sdk_type)


def request_id_from_exception(exc: Exception) -> str | None:
    return getattr(exc, "request_id", None) or getattr(exc, "_request_id", None)


def map_openai_compatible_exception(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    name = type(exc).__name__
    kwargs = {"provider_request_id": request_id_from_exception(exc)}
    if is_sdk_exception(exc, RateLimitError) or "RateLimit" in name:
        return ProviderError.from_code(FailureCode.RATE_LIMITED, **kwargs)
    if is_sdk_exception(exc, APITimeoutError) or "Timeout" in name:
        return ProviderError.from_code(FailureCode.PROVIDER_TIMEOUT, **kwargs)
    status = getattr(exc, "status_code", None)
    if status == 429:
        return ProviderError.from_code(FailureCode.RATE_LIMITED, **kwargs)
    if isinstance(status, int) and status >= 500:
        return ProviderError.from_code(FailureCode.PROVIDER_UNAVAILABLE, **kwargs)
    if is_sdk_exception(exc, APIError):
        return ProviderError.from_code(FailureCode.PROVIDER_ERROR, **kwargs)
    return ProviderError.from_code(FailureCode.PROVIDER_ERROR, **kwargs)
