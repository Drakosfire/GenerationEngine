"""Image generation service that orchestrates providers and uploads."""

import time
from datetime import datetime
from typing import Dict

from generationengine.models.errors import ErrorCode, GenerationError
from generationengine.models.image_responses import ImageGenerationResponse, ImageResult
from generationengine.models.metrics import GenerationMetrics
from generationengine.models.requests import ImageGenerationRequest, ImageModel
from generationengine.providers.base import ImageProvider
from generationengine.providers.fal_provider import FalProvider
from generationengine.providers.openai_provider import OpenAIImageProvider
from generationengine.services.retry_service import retry_with_backoff, RetryableError, should_retry
from generationengine.services.upload_service import UploadService
import json


class ImageService:
    """Unified service for image generation across all providers."""

    def __init__(
        self,
        fal_api_key: str | None = None,
        openai_api_key: str | None = None,
        upload_service: UploadService | None = None,
    ):
        """
        Initialize image service with providers.

        Args:
            fal_api_key: Fal.ai API key (defaults to FAL_KEY env var)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            upload_service: Upload service instance (creates new one if not provided)
        """
        # Initialize providers
        self.providers: Dict[str, ImageProvider] = {}

        try:
            self.providers["flux-pro"] = FalProvider(api_key=fal_api_key)
            self.providers["imagen4"] = FalProvider(api_key=fal_api_key)
            self.providers["imagen4-fast"] = FalProvider(api_key=fal_api_key)
        except (ImportError, ValueError) as e:
            # Fal provider not available - skip it
            pass

        try:
            self.providers["openai"] = OpenAIImageProvider(api_key=openai_api_key)
        except (ImportError, ValueError) as e:
            # OpenAI provider not available - skip it
            pass

        # Initialize upload service
        self.upload_service = upload_service or UploadService()

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate images using the specified model and upload to Cloudflare.

        Args:
            request: Image generation request

        Returns:
            ImageGenerationResponse with generated images or error
        """
        start_time = time.time()
        retry_count = 0

        # Serialize input for metrics
        input_json = json.dumps({
            "prompt": request.prompt,
            "model": request.model.value,
            "num_images": request.num_images,
            "size": request.size.value,
        })

        try:
            # Determine provider based on model
            model_key = request.model.value
            provider = self.providers.get(model_key)

            if not provider:
                return ImageGenerationResponse(
                    success=False,
                    error=GenerationError(
                        code=ErrorCode.INVALID_INPUT,
                        message=f"Unsupported model: {model_key}. Available: {list(self.providers.keys())}",
                        retryable=False,
                    ),
                )

            # Generate images with retry logic
            try:
                image_bytes_list = await retry_with_backoff(
                    provider.generate,
                    request.prompt,
                    model_key,
                    request.num_images,
                    request.get_size_tuple(),
                )
            except RetryableError as e:
                # Retry exhausted - return error
                retry_count = 3  # Max retries attempted
                raise e
            except Exception as e:
                # Unexpected error - wrap it
                raise RetryableError(
                    ErrorCode.INTERNAL_ERROR,
                    f"Generation failed: {str(e)}",
                    original_exception=e,
                )

            # Upload all images to Cloudflare (in parallel would be better, but keep it simple for now)
            image_results: list[ImageResult] = []
            size_tuple = request.get_size_tuple()

            for idx, image_bytes in enumerate(image_bytes_list):
                try:
                    url = await self.upload_service.upload_image(
                        image_bytes,
                        prefix="generated",
                        filename=f"generated_{model_key}_{idx}.png",
                    )

                    image_results.append(
                        ImageResult(
                            url=url,
                            width=size_tuple[0],
                            height=size_tuple[1],
                            model_used=model_key,
                        )
                    )
                except Exception as upload_error:
                    # If upload fails, skip this image but continue with others
                    # In production, you might want to retry or handle differently
                    continue

            if not image_results:
                # All uploads failed
                return ImageGenerationResponse(
                    success=False,
                    error=GenerationError(
                        code=ErrorCode.INTERNAL_ERROR,
                        message="Image generation succeeded but all uploads failed",
                        retryable=True,
                    ),
                )

            # Calculate metrics
            duration_ms = int((time.time() - start_time) * 1000)
            output_json = json.dumps({
                "image_count": len(image_results),
                "uploaded_count": len(image_results),
            })

            metrics = GenerationMetrics(
                duration_ms=duration_ms,
                model_used=model_key,
                retry_count=retry_count,
                timestamp=datetime.now(),
                input=input_json,
                output=output_json,
            )

            return ImageGenerationResponse(
                success=True,
                images=image_results,
                metrics=metrics,
            )

        except RetryableError as e:
            # Retryable error - return error response
            duration_ms = int((time.time() - start_time) * 1000)

            return ImageGenerationResponse(
                success=False,
                error=GenerationError(
                    code=e.error_code,
                    message=e.message,
                    retryable=should_retry(e.error_code),
                ),
                metrics=GenerationMetrics(
                    duration_ms=duration_ms,
                    retry_count=retry_count,
                    timestamp=datetime.now(),
                    input=input_json,
                ),
            )
        except Exception as e:
            # Unexpected error
            duration_ms = int((time.time() - start_time) * 1000)

            return ImageGenerationResponse(
                success=False,
                error=GenerationError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Unexpected error: {str(e)}",
                    retryable=False,
                ),
                metrics=GenerationMetrics(
                    duration_ms=duration_ms,
                    retry_count=retry_count,
                    timestamp=datetime.now(),
                    input=input_json,
                ),
            )

