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
    ) -> list[bytes]:
        """
        Generate images from a prompt.

        Args:
            prompt: Text prompt for image generation
            model: Model identifier (e.g., "flux-pro", "imagen4")
            num_images: Number of images to generate (1-8)
            size: Output size as (width, height) tuple

        Returns:
            List of image bytes (one per generated image)

        Raises:
            Exception: Provider-specific errors
        """
        ...

