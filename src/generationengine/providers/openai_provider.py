"""OpenAI image generation provider."""

import base64
import os
from typing import Tuple

try:
    from openai import OpenAI
    from openai import RateLimitError, APITimeoutError, APIError
except ImportError:
    OpenAI = None  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIError = Exception  # type: ignore

from generationengine.models.errors import ErrorCode
from generationengine.providers.base import ImageProvider
from generationengine.services.retry_service import RetryableError


class OpenAIImageProvider:
    """Image provider using OpenAI Images API."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        """
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required")

        self.client = OpenAI(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        model: str,
        num_images: int,
        size: Tuple[int, int],
    ) -> list[bytes]:
        """
        Generate images using OpenAI Images API.

        Args:
            prompt: Text prompt for image generation
            model: Model identifier (should be "openai" or "gpt-image-1-mini")
            num_images: Number of images to generate (1-8)
            size: Output size as (width, height) tuple

        Returns:
            List of image bytes (one per generated image)

        Raises:
            RetryableError: For retryable failures (timeouts, rate limits)
            Exception: For non-retryable failures
        """
        # Map size tuple to OpenAI size string format
        size_str = f"{size[0]}x{size[1]}"

        # OpenAI models
        openai_model = "gpt-image-1-mini" if model == "openai" or "mini" in model.lower() else "dall-e-3"

        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            loop = asyncio.get_event_loop()

            # OpenAI SDK is synchronous, so we run it in a thread pool
            with ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    lambda: self.client.images.generate(
                        model=openai_model,
                        prompt=prompt,
                        n=num_images,
                        size=size_str,
                        response_format="b64_json",  # Request base64 encoded images
                    ),
                )

            # Decode base64 images to bytes
            images: list[bytes] = []
            for image_data in response.data:
                if hasattr(image_data, "b64_json") and image_data.b64_json:
                    image_bytes = base64.b64decode(image_data.b64_json)
                    images.append(image_bytes)
                elif hasattr(image_data, "url") and image_data.url:
                    # Fallback: download from URL if base64 not available
                    import httpx

                    async with httpx.AsyncClient() as client:
                        img_response = await client.get(image_data.url)
                        img_response.raise_for_status()
                        images.append(img_response.content)
                else:
                    # Skip invalid image data
                    continue

            return images

        except RateLimitError as e:
            raise RetryableError(
                ErrorCode.RATE_LIMITED,
                f"OpenAI rate limit exceeded: {str(e)}",
                original_exception=e,
            )
        except APITimeoutError as e:
            raise RetryableError(
                ErrorCode.PROVIDER_TIMEOUT,
                f"OpenAI request timed out: {str(e)}",
                original_exception=e,
            )
        except APIError as e:
            # Check if it's a retryable error
            if hasattr(e, "status_code"):
                if e.status_code == 429:
                    raise RetryableError(
                        ErrorCode.RATE_LIMITED,
                        f"OpenAI rate limit exceeded: {str(e)}",
                        original_exception=e,
                    )
                if e.status_code >= 500:
                    raise RetryableError(
                        ErrorCode.PROVIDER_OVERLOADED,
                        f"OpenAI server error {e.status_code}: {str(e)}",
                        original_exception=e,
                    )

            # Non-retryable error (4xx client errors)
            raise RetryableError(
                ErrorCode.PROVIDER_REJECTED,
                f"OpenAI API error: {str(e)}",
                original_exception=e,
            )
        except Exception as e:
            raise RetryableError(
                ErrorCode.INTERNAL_ERROR,
                f"OpenAI generation failed: {str(e)}",
                original_exception=e,
            )

