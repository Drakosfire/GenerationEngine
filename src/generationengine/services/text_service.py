"""Text generation service with retry logic and error handling.

Supports OpenAI Structured Outputs for guaranteed schema-compliant responses.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

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
from generationengine.utils.schema_utils import make_schema_strict

logger = logging.getLogger(__name__)


# Pricing per 1K tokens (as of 2024, approximate values)
# Source: https://openai.com/api/pricing/
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.1": {"prompt": 0.005, "completion": 0.015},  # Estimated, update when pricing available
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

    def __init__(
        self,
        openai_api_key: str | None = None,
        metrics_service: Any | None = None,  # Type is 'Any' to avoid circular import
    ):
        """
        Initialize text generation service.

        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            metrics_service: Optional MetricsService for recording metrics
        """
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass openai_api_key.")
        
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self._metrics_service = metrics_service

    async def generate(
        self,
        request: TextGenerationRequest,
        service_name: str | None = None,
    ) -> TextGenerationResponse:
        """
        Generate text using OpenAI with retry logic and error handling.

        Args:
            request: Text generation request
            service_name: Optional service name for metrics categorization

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
            # Build request kwargs for Responses API
            request_kwargs: dict[str, Any] = {
                "model": request.model.value,
                "input": request.user_prompt,
                "temperature": request.temperature,
            }
            
            # Add instructions (system prompt) if provided
            if request.system_prompt:
                request_kwargs["instructions"] = request.system_prompt
            
            # Note: max_tokens is NOT supported in Responses API (non-streaming or streaming)
            # The Responses API doesn't have a token limit parameter
            if request.max_tokens:
                logger.warning("📋 [TextService] max_tokens not supported in Responses API, ignoring")

            # Add structured output schema if provided (Responses API format)
            if request.response_schema:
                strict_schema = make_schema_strict(request.response_schema)
                request_kwargs["text"] = {
                    "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.response_schema_name,
                        "strict": True,
                        "schema": strict_schema,
                        }
                    }
                }
                logger.info(f"📋 [TextService] Using structured outputs with schema: {request.response_schema_name}")

            # Call OpenAI with retry logic
            response = await retry_with_backoff(
                self._call_openai,
                **request_kwargs,
                timeout_seconds=60.0,  # 60s timeout per attempt
            )

            # Check for refusal (can happen with structured outputs)
            # Responses API may have refusal in different format - check if needed
            if hasattr(response, "refusal") and response.refusal:
                logger.warning(f"🚫 [TextService] OpenAI refused generation: {response.refusal}")
                return TextGenerationResponse(
                    success=False,
                    error=GenerationError(
                        code=ErrorCode.PROVIDER_REJECTED,
                        message=f"Generation refused: {response.refusal}",
                        retryable=False,
                    ),
                    metrics=GenerationMetrics(
                        duration_ms=int((time.time() - start_time) * 1000),
                        retry_count=retry_count,
                        timestamp=datetime.now(timezone.utc),
                        input=input_json,
                    ),
                )

            # Success - extract content and usage (Responses API format)
            content = response.output_text if hasattr(response, "output_text") else None
            usage = response.usage if hasattr(response, "usage") else None

            # Calculate metrics
            # Responses API uses input_tokens/output_tokens, not prompt_tokens/completion_tokens
            duration_ms = int((time.time() - start_time) * 1000)
            prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
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

            # Parse JSON when using structured outputs
            parsed_content = None
            structured_output = request.response_schema is not None
            if structured_output and content:
                try:
                    parsed_content = json.loads(content)
                    logger.info(f"✅ [TextService] Structured output parsed successfully")
                except json.JSONDecodeError as e:
                    # This should never happen with structured outputs, but handle gracefully
                    logger.error(f"❌ [TextService] Failed to parse structured output: {e}")

            # Record metrics if service available
            if self._metrics_service and hasattr(self._metrics_service, 'record'):
                self._metrics_service.record(metrics, service_name)

            return TextGenerationResponse(
                success=True,
                content=content,
                parsed_content=parsed_content,
                structured_output=structured_output,
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

    async def generate_stream(
        self,
        request: TextGenerationRequest,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using OpenAI Responses API with streaming support.
        
        Args:
            request: Text generation request
            
        Yields:
            str: Text chunks in SSE format ("data: {content}\n\n")
        """
        try:
            # Build request kwargs for Responses API streaming
            # Note: max_tokens is NOT supported in Responses API streaming mode
            # Ensure all values are plain Python types (str, float) for SDK compatibility
            request_kwargs: dict[str, Any] = {
                "model": str(request.model.value),  # Ensure string type
                "input": str(request.user_prompt),  # Ensure string type
                "temperature": float(request.temperature),  # Ensure float type
            }
            
            # Add instructions (system prompt) if provided
            if request.system_prompt:
                request_kwargs["instructions"] = str(request.system_prompt)  # Ensure string type
            
            # Note: max_tokens is not supported in Responses API streaming
            if request.max_tokens:
                logger.warning("📋 [TextService] max_tokens not supported in Responses API streaming mode, ignoring")

            # Note: Structured outputs (response_schema) not supported in streaming mode
            # If structured outputs are needed, use generate() instead
            if request.response_schema:
                logger.warning("📋 [TextService] Structured outputs not supported in streaming mode, ignoring schema")

            # Create streaming manager
            # Ensure all values are JSON-serializable (strings, numbers, None)
            # The SDK may internally serialize these, so we need plain Python types
            stream_manager = self.openai_client.responses.stream(**request_kwargs)
            
            # Use async context manager for stream
            async with stream_manager as response_stream:
                async for event in response_stream:
                    event_type = getattr(event, "type", None)
                    
                    if event_type == "response.output_text.delta":
                        content = getattr(event, "delta", "")
                        if content:
                            # Yield in SSE format
                            yield f"data: {content}\n\n"
                    
                    elif event_type == "response.error":
                        error_message = getattr(getattr(event, "error", None), "message", "Unknown Responses error")
                        logger.error(f"❌ [TextService] Responses error event: {error_message}")
                        yield f"data: [ERROR]{error_message}\n\n"
                        return
                    
                    elif event_type == "response.completed":
                        logger.debug("✅ [TextService] Responses stream completed")
            
            # Send completion signal
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"❌ [TextService] Streaming error: {str(e)}", exc_info=True)
            yield f"data: [ERROR]{str(e)}\n\n"

    async def _call_openai(self, **kwargs: Any) -> Any:
        """
        Internal OpenAI API call that raises RetryableError on retryable failures.

        Args:
            **kwargs: Arguments for OpenAI responses.create

        Returns:
            OpenAI response object

        Raises:
            RetryableError: For retryable failures (timeouts, rate limits)
            Exception: For non-retryable errors (re-raised as-is)
        """
        try:
            return await self.openai_client.responses.create(**kwargs)
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

