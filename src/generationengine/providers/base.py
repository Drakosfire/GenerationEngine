"""Base provider interface for image generation."""

from typing import Protocol, Tuple

from typing_extensions import runtime_checkable


@runtime_checkable
class ImageProvider(Protocol):
    """Protocol for image generation providers."""

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
        Generate images from a prompt.

        Args:
            prompt: Text prompt for image generation
            model: Model identifier (e.g., "flux-2-pro", "nano-banana-pro", "gpt-image-1.5")
            num_images: Number of images to generate (1-8)
            size: Output size as (width, height) tuple
            image_url: Optional source image URL for image-to-image generation
            strength: Transformation strength for image-to-image (0.0-1.0)
            mask_base64: Optional base64-encoded PNG mask for inpainting
            base_image_base64: Optional base64-encoded PNG base image for inpainting
            negative_prompt: Elements to exclude from generation (e.g., "grid, text")

        Returns:
            List of image bytes (one per generated image)

        Raises:
            Exception: Provider-specific errors
        """
        ...

