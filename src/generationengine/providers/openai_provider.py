"""OpenAI image generation provider."""

import base64
import os
from typing import Tuple

try:
    from openai import APIError, APITimeoutError, OpenAI, RateLimitError
except ImportError:
    OpenAI = None  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIError = Exception  # type: ignore

from generationengine.models.errors import ErrorCode
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
        image_url: str | None = None,
        strength: float | None = None,
        mask_base64: str | None = None,
        base_image_base64: str | None = None,
        negative_prompt: str | None = None,
    ) -> list[bytes]:
        """
        Generate images using OpenAI Images API.

        Args:
            prompt: Text prompt for image generation
            model: Model identifier (should be "openai" or "gpt-image-1-mini")
            num_images: Number of images to generate (1-8)
            size: Output size as (width, height) tuple
            image_url: Ignored - OpenAI doesn't support image-to-image via this API
            strength: Ignored - OpenAI doesn't support image-to-image via this API
            mask_base64: Ignored - OpenAI inpainting requires images.edit endpoint
            base_image_base64: Ignored - OpenAI inpainting requires images.edit endpoint
            negative_prompt: Ignored - OpenAI native API doesn't support negative prompts

        Returns:
            List of image bytes (one per generated image)

        Raises:
            RetryableError: For retryable failures (timeouts, rate limits)
            Exception: For non-retryable failures

        Note:
            image_url, strength, mask_base64, base_image_base64, and negative_prompt
            are accepted for API compatibility with other providers but are ignored.
            OpenAI image-to-image requires a different endpoint (images.edit).
        """
        # Warn if image-to-image params are provided (not supported by this provider)
        if image_url:
            import logging
            logging.getLogger(__name__).warning(
                "OpenAI provider does not support image-to-image generation. "
                "image_url parameter will be ignored."
            )
        # OpenAI models - determine which API to use
        is_gpt_image = model == "openai" or "gpt-image" in model.lower() or "mini" in model.lower()
        openai_model = "gpt-image-1-mini" if is_gpt_image else "dall-e-3"

        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            import httpx

            loop = asyncio.get_event_loop()

            # OpenAI SDK is synchronous, so we run it in a thread pool
            with ThreadPoolExecutor() as executor:
                if is_gpt_image:
                    # GPT-Image API (gpt-image-1, gpt-image-1-mini, gpt-image-1.5)
                    # - Does NOT support response_format parameter
                    # - Returns b64_json by default
                    # - Sizes: 1024x1024 (square), 1536x1024 (landscape), 1024x1536 (portrait)
                    response = await loop.run_in_executor(
                        executor,
                        lambda: self.client.images.generate(
                            model=openai_model,
                            prompt=prompt,
                            n=num_images,
                            size="1024x1024",  # GPT-Image supports: 1024x1024, 1536x1024, 1024x1536
                        ),
                    )
                else:
                    # DALL-E API (dall-e-2, dall-e-3)
                    # - Supports response_format
                    # - Can return base64 or URL
                    size_str = f"{size[0]}x{size[1]}"
                    response = await loop.run_in_executor(
                        executor,
                        lambda: self.client.images.generate(
                            model=openai_model,
                            prompt=prompt,
                            n=num_images,
                            size=size_str,
                            response_format="b64_json",
                        ),
                    )

            # Extract images from response
            images: list[bytes] = []
            for image_data in response.data:
                # Try base64 first (DALL-E with response_format)
                if hasattr(image_data, "b64_json") and image_data.b64_json:
                    image_bytes = base64.b64decode(image_data.b64_json)
                    images.append(image_bytes)
                # Fall back to URL (GPT-Image or DALL-E with url format)
                elif hasattr(image_data, "url") and image_data.url:
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

