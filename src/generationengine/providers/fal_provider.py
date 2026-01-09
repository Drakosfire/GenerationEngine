"""Fal.ai image generation provider."""

import logging
import os
from typing import Tuple

try:
    import fal_client
except ImportError:
    fal_client = None  # type: ignore

import httpx

from generationengine.models.errors import ErrorCode
from generationengine.services.retry_service import RetryableError

logger = logging.getLogger(__name__)


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

    def _size_to_gpt_image_size(self, size: Tuple[int, int]) -> str:
        """
        Convert (width, height) to GPT Image 1.5 size string.

        Supported sizes:
        - "1024x1024" (square)
        - "1536x1024" (landscape)
        - "1024x1536" (portrait)
        """
        width, height = size

        # Direct mapping for exact matches
        size_map = {
            (1024, 1024): "1024x1024",
            (1536, 1024): "1536x1024",
            (1024, 1536): "1024x1536",
        }

        if (width, height) in size_map:
            return size_map[(width, height)]

        # Fallback: determine closest match based on aspect ratio
        if width > height:
            return "1536x1024"  # Landscape
        elif height > width:
            return "1024x1536"  # Portrait
        else:
            return "1024x1024"  # Square

    async def generate(
        self,
        prompt: str,
        model: str,
        num_images: int,
        size: Tuple[int, int],
        image_url: str | None = None,
        strength: float = 0.85,
        mask_base64: str | None = None,
        base_image_base64: str | None = None,
        negative_prompt: str | None = None,
    ) -> list[bytes]:
        """
        Generate images using Fal.ai.

        Args:
            prompt: Text prompt for image generation
            model: Model identifier (flux-2-pro, nano-banana-pro, gpt-image-1.5)
            num_images: Number of images to generate (1-8)
            size: Output size as (width, height) tuple
            image_url: Optional source image URL for image-to-image generation
            strength: Transformation strength for image-to-image (0.0-1.0), default 0.85
            mask_base64: Optional base64-encoded mask for inpainting (GPT Image 1.5 only)
            base_image_base64: Base64-encoded base image for inpainting (/edit endpoint)
            negative_prompt: Elements to exclude from generation

        Returns:
            List of image bytes (one per generated image)

        Raises:
            RetryableError: For retryable failures (timeouts, rate limits)
            Exception: For non-retryable failures
        """
        # Map model names to Fal endpoints (text-to-image)
        model_endpoints = {
            "flux-2-pro": "fal-ai/flux-2-pro",
            "nano-banana-pro": "fal-ai/nano-banana-pro",
            "gpt-image-1.5": "fal-ai/gpt-image-1.5",
            "flux-pro": "fal-ai/flux-pro",
            "flux-lora-i2i": "fal-ai/flux-lora/image-to-image",
        }

        # Map model names to edit endpoints (inpainting)
        edit_endpoints = {
            "flux-2-pro": "fal-ai/flux-2-pro/edit",
            "nano-banana-pro": "fal-ai/nano-banana-pro/edit",
            "gpt-image-1.5": "fal-ai/gpt-image-1.5/edit",
        }

        endpoint = model_endpoints.get(model)
        if not endpoint:
            raise ValueError(f"Unsupported Fal model: {model}")
        
        # Switch to edit endpoint for inpainting (when base image provided)
        is_inpainting = base_image_base64 is not None
        if is_inpainting:
            edit_endpoint = edit_endpoints.get(model)
            if edit_endpoint:
                endpoint = edit_endpoint
            else:
                raise ValueError(f"Model {model} does not support inpainting")

        # Build arguments based on model and whether inpainting
        if is_inpainting:
            # All edit endpoints use image_urls array
            # GPT Image 1.5 also requires mask_image_url
            if model == "gpt-image-1.5":
                size_str = self._size_to_gpt_image_size(size)
                arguments = {
                    "prompt": prompt,
                    "image_urls": [base_image_base64],
                    "mask_image_url": mask_base64,  # GPT Image 1.5 requires explicit mask
                    "num_images": num_images,
                    "image_size": size_str,
                    "quality": "high",
                    "output_format": "png",
                    "background": "opaque",
                }
            elif model == "nano-banana-pro":
                aspect_ratio = self._size_to_aspect_ratio(size)
                arguments = {
                    "prompt": prompt,
                    "image_urls": [base_image_base64],
                    "num_images": num_images,
                    "aspect_ratio": aspect_ratio,
                    "output_format": "png",
                }
                if negative_prompt:
                    arguments["negative_prompt"] = negative_prompt
            else:  # flux-2-pro
                arguments = {
                    "prompt": prompt,
                    "image_urls": [base_image_base64],
                    "num_images": num_images,
                    "image_size": {
                        "width": size[0],
                        "height": size[1],
                    },
                }
                if negative_prompt:
                    arguments["negative_prompt"] = negative_prompt
        else:
            # Text-to-image generation
            if model == "gpt-image-1.5":
                size_str = self._size_to_gpt_image_size(size)
                arguments = {
                    "prompt": prompt,
                    "num_images": num_images,
                    "image_size": size_str,
                    "quality": "high",
                    "output_format": "png",
                    "background": "opaque",
                }
                if negative_prompt:
                    arguments["negative_prompt"] = negative_prompt
            elif model == "nano-banana-pro":
                aspect_ratio = self._size_to_aspect_ratio(size)
                arguments = {
                    "prompt": prompt,
                    "num_images": num_images,
                    "aspect_ratio": aspect_ratio,
                    "output_format": "png",
                }
                if negative_prompt:
                    arguments["negative_prompt"] = negative_prompt
            else:  # flux-2-pro
                arguments = {
                    "prompt": prompt,
                    "num_images": num_images,
                    "image_size": {
                        "width": size[0],
                        "height": size[1],
                    },
                    "enable_safety_checker": True,
                }
                if negative_prompt:
                    arguments["negative_prompt"] = negative_prompt

        # Add image-to-image parameters if image_url is provided (not inpainting)
        if image_url and not is_inpainting:
            arguments["image_url"] = image_url
            arguments["strength"] = strength
            # flux-lora/image-to-image uses additional parameters
            if model == "flux-lora-i2i":
                arguments["num_inference_steps"] = 35

        # Log request details
        logger.info(f"🎨 [FalProvider] Calling endpoint: {endpoint}")
        logger.info(f"🎨 [FalProvider] Model: {model}, num_images: {num_images}, size: {size}")
        
        # Log inpainting details if present
        if is_inpainting:
            base_size_kb = len(base_image_base64) / 1024  # type: ignore
            logger.info(f"🖼️ [FalProvider] INPAINTING MODE: base_image={base_size_kb:.1f}KB")
            if mask_base64:
                mask_size_kb = len(mask_base64) / 1024
                logger.info(f"🖼️ [FalProvider] Mask provided: {mask_size_kb:.1f}KB")
            logger.info(f"🖼️ [FalProvider] Using {model} EDIT endpoint with image_urls array")
        
        # Log negative prompt if present
        if negative_prompt:
            logger.info(f"🚫 [FalProvider] Negative prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
        
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
                logger.warning(f"⚠️ [FalProvider] Partial success: got {len(images)}/{num_images} images")

            logger.info(f"✅ [FalProvider] Generation complete: {len(images)} images downloaded")
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

