"""Fal.ai image generation provider."""

import os
from typing import Tuple

try:
    import fal_client
except ImportError:
    fal_client = None  # type: ignore

import httpx

from generationengine.models.errors import ErrorCode
from generationengine.services.retry_service import RetryableError


class FalProvider:
    """Image provider using Fal.ai API."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize Fal provider.

        Args:
            api_key: Fal.ai API key (defaults to FAL_KEY environment variable)
        """
        if fal_client is None:
            raise ImportError(
                "fal-client package is required. Install with: pip install fal-client"
            )

        self.api_key = api_key or os.getenv("FAL_KEY")
        if not self.api_key:
            raise ValueError("FAL_KEY environment variable or api_key parameter is required")

        # Set API key for fal_client
        os.environ["FAL_KEY"] = self.api_key

    def _size_to_aspect_ratio(self, size: Tuple[int, int]) -> str:
        """
        Convert (width, height) to aspect ratio string.

        Common mappings:
        - 1024x1024 -> "1:1"
        - 1536x1024 -> "3:2" (landscape)
        - 1024x1536 -> "2:3" (portrait)
        - 1792x1024 -> "16:9" (wide)
        - 2016x1024 -> "21:9" (ultra-wide)
        """
        from math import gcd

        width, height = size
        divisor = gcd(width, height)
        ratio_w = width // divisor
        ratio_h = height // divisor

        # Simplify common ratios to standard formats
        ratio_map = {
            (1, 1): "1:1",
            (3, 2): "3:2",
            (2, 3): "2:3",
            (4, 3): "4:3",
            (3, 4): "3:4",
            (16, 9): "16:9",
            (9, 16): "9:16",
            (21, 9): "21:9",
            (9, 21): "9:21",
        }

        return ratio_map.get((ratio_w, ratio_h), f"{ratio_w}:{ratio_h}")

    async def generate(
        self,
        prompt: str,
        model: str,
        num_images: int,
        size: Tuple[int, int],
        image_url: str | None = None,
        strength: float = 0.85,
    ) -> list[bytes]:
        """
        Generate images using Fal.ai.

        Args:
            prompt: Text prompt for image generation
            model: Model identifier (flux-pro, imagen4, imagen4-fast, flux-lora-i2i)
            num_images: Number of images to generate (1-8)
            size: Output size as (width, height) tuple
            image_url: Optional source image URL for image-to-image generation
            strength: Transformation strength for image-to-image (0.0-1.0), default 0.85

        Returns:
            List of image bytes (one per generated image)

        Raises:
            RetryableError: For retryable failures (timeouts, rate limits)
            Exception: For non-retryable failures
        """
        # Map model names to Fal endpoints
        model_endpoints = {
            "flux-pro": "fal-ai/flux-pro/new",
            "flux-lora-i2i": "fal-ai/flux-lora/image-to-image",
            "nano-banana": "fal-ai/nano-banana",
            "hunyuan": "fal-ai/hunyuan-image/v3/text-to-image",
            "dreamina": "fal-ai/bytedance/dreamina/v3.1/text-to-image",
            "flux-kontext": "fal-ai/flux-pro/kontext/max/text-to-image",
        }

        endpoint = model_endpoints.get(model)
        if not endpoint:
            raise ValueError(f"Unsupported Fal model: {model}")

        # Models that use aspect_ratio instead of image_size
        aspect_ratio_models = {"nano-banana"}

        # Build arguments dict
        if model in aspect_ratio_models:
            # Nano-banana and similar models use aspect_ratio
            aspect_ratio = self._size_to_aspect_ratio(size)
            arguments = {
                "prompt": prompt,
                "num_images": num_images,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
            }
        else:
            # Standard models use image_size
            arguments = {
                "prompt": prompt,
                "num_images": num_images,
                "image_size": {
                    "width": size[0],
                    "height": size[1],
                },
                "enable_safety_checker": True,
            }

        # Add image-to-image parameters if image_url is provided
        if image_url:
            arguments["image_url"] = image_url
            arguments["strength"] = strength
            # flux-lora/image-to-image also uses num_inference_steps
            if model == "flux-lora-i2i":
                arguments["num_inference_steps"] = 35

        try:
            # Use subscribe_async for non-blocking async operation
            # IMPORTANT: fal_client.subscribe() is BLOCKING and will freeze the event loop!
            fal_result = await fal_client.subscribe_async(
                endpoint,
                arguments=arguments,
            )

            if not fal_result or "images" not in fal_result:
                raise ValueError("Fal.ai returned empty result")

            # Download images from URLs
            images: list[bytes] = []
            async with httpx.AsyncClient() as client:
                for image_data in fal_result["images"]:
                    image_url = image_data.get("url")
                    if not image_url:
                        continue

                    response = await client.get(image_url)
                    response.raise_for_status()
                    images.append(response.content)

            if len(images) < num_images:
                # Partial success - still return what we got
                pass

            return images

        except (TimeoutError, httpx.TimeoutException) as e:
            raise RetryableError(
                ErrorCode.PROVIDER_TIMEOUT,
                f"Fal.ai request timed out: {str(e)}",
                original_exception=e,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RetryableError(
                    ErrorCode.RATE_LIMITED,
                    f"Fal.ai rate limit exceeded: {str(e)}",
                    original_exception=e,
                )
            raise RetryableError(
                ErrorCode.PROVIDER_OVERLOADED,
                f"Fal.ai returned error {e.response.status_code}: {str(e)}",
                original_exception=e,
            )
        except Exception as e:
            # Wrap unexpected errors as internal errors
            raise RetryableError(
                ErrorCode.INTERNAL_ERROR,
                f"Fal.ai generation failed: {str(e)}",
                original_exception=e,
            )

