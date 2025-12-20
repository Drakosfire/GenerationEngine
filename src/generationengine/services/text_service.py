"""Text generation service with retry logic and error handling."""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

try:
    from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError
except ImportError:
    # OpenAI not available - define stubs for type checking
    AsyncOpenAI = object  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIError = Exception  # type: ignore

from generationengine.models.errors import ErrorCode, is_retryable
from generationengine.models.metrics import GenerationMetrics
from generationengine.models.requests import TextGenerationRequest
from generationengine.models.responses import GenerationError
from generationengine.models.text_responses import TextGenerationResponse
from generationengine.services.retry_service import RetryableError, retry_with_backoff


# Pricing per 1K tokens (as of 2024, approximate values)
# Source: https://openai.com/api/pricing/
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-2024-08-06": {"prompt": 0.005, "completion": 0.015},
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for OpenAI API call."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0
    
    prompt_cost = (prompt_tokens / 1000.0) * pricing["prompt"]
    completion_cost = (completion_tokens / 1000.0) * pricing["completion"]
    return prompt_cost + completion_cost


class TextGenerationService:
    """Unified service for text generation with retry logic and error handling."""

    def __init__(self, openai_api_key: str | None = None):
        """
        Initialize text generation service.

        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass openai_api_key.")
        
        self.openai_client = AsyncOpenAI(api_key=api_key)

    async def generate(self, request: TextGenerationRequest) -> TextGenerationResponse:
        """
        Generate text using OpenAI with retry logic and error handling.

        Args:
            request: Text generation request

        Returns:
            TextGenerationResponse with generated content or error
        """
        start_time = time.time()
        retry_count = 0

        # Serialize input for metrics
        input_json = json.dumps({
            "system_prompt": request.system_prompt,
            "user_prompt_length": len(request.user_prompt),
            "model": request.model.value,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        })

        try:
            # Build messages
            messages: list[dict[str, str]] = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.user_prompt})

            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": request.model.value,
                "messages": messages,
                "temperature": request.temperature,
            }
            if request.max_tokens:
                request_kwargs["max_tokens"] = request.max_tokens

            # Call OpenAI with retry logic
            response = await retry_with_backoff(
                self._call_openai,
                **request_kwargs,
                timeout_seconds=60.0,  # 60s timeout per attempt
            )

            # Success - extract content and usage
            content = response.choices[0].message.content
            usage = response.usage

            # Calculate metrics
            duration_ms = int((time.time() - start_time) * 1000)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            estimated_cost = estimate_cost(request.model.value, prompt_tokens, completion_tokens)

            metrics = GenerationMetrics(
                duration_ms=duration_ms,
                tokens_used=total_tokens,
                estimated_cost_usd=estimated_cost,
                model_used=request.model.value,
                retry_count=retry_count,
                    timestamp=datetime.now(timezone.utc),
                input=input_json,
                output=json.dumps({
                    "content_length": len(content) if content else 0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }),
            )

            return TextGenerationResponse(
                success=True,
                content=content,
                metrics=metrics,
            )

        except RetryableError as e:
            # Retry exhausted - map to appropriate error
            retry_count = 3  # Max retries attempted
            error_code = e.error_code
            error_message = e.message
            retryable = True  # Already retried, but mark as retryable for client awareness

        except Exception as e:
            # Unexpected error - wrap it
            error_code = ErrorCode.INTERNAL_ERROR
            error_message = f"Text generation failed: {str(e)}"
            retryable = False

        # Return error response
        duration_ms = int((time.time() - start_time) * 1000)
        metrics = GenerationMetrics(
            duration_ms=duration_ms,
            retry_count=retry_count,
                    timestamp=datetime.now(timezone.utc),
            input=input_json,
        )

        error = GenerationError(
            code=error_code,
            message=error_message,
            retryable=retryable if 'retryable' in locals() else is_retryable(error_code),
        )

        return TextGenerationResponse(
            success=False,
            error=error,
            metrics=metrics,
        )

    async def _call_openai(self, **kwargs: Any) -> Any:
        """
        Internal OpenAI API call that raises RetryableError on retryable failures.

        Args:
            **kwargs: Arguments for OpenAI chat.completions.create

        Returns:
            OpenAI response object

        Raises:
            RetryableError: For retryable failures (timeouts, rate limits)
            Exception: For non-retryable errors (re-raised as-is)
        """
        try:
            return await self.openai_client.chat.completions.create(**kwargs)
        except APITimeoutError as e:
            raise RetryableError(
                ErrorCode.PROVIDER_TIMEOUT,
                f"OpenAI API request timed out: {str(e)}",
                original_exception=e,
            )
        except RateLimitError as e:
            raise RetryableError(
                ErrorCode.RATE_LIMITED,
                f"OpenAI API rate limit exceeded: {str(e)}",
                original_exception=e,
            )
        except APIError as e:
            # Check if it's a retryable error based on status code
            status_code = getattr(e, "status_code", None)
            if status_code in [429, 500, 502, 503, 504]:
                raise RetryableError(
                    ErrorCode.PROVIDER_OVERLOADED,
                    f"OpenAI API error (retryable): {str(e)}",
                    original_exception=e,
                )
            else:
                # Non-retryable API error
                raise RetryableError(
                    ErrorCode.PROVIDER_REJECTED,
                    f"OpenAI API error: {str(e)}",
                    original_exception=e,
                )
        except Exception as e:
            # Unexpected error - wrap as internal error
            raise RetryableError(
                ErrorCode.INTERNAL_ERROR,
                f"Unexpected error during OpenAI call: {str(e)}",
                original_exception=e,
            )

