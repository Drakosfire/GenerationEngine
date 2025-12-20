"""Tests for Fal image provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from generationengine.providers.fal_provider import FalProvider
from generationengine.models.errors import ErrorCode


@pytest.mark.asyncio
async def test_fal_provider_generate_success():
    """Test successful image generation with Fal provider."""
    with patch("generationengine.providers.fal_provider.fal_client") as mock_fal_client:
        # Mock Fal response
        mock_result = {
            "images": [
                {"url": "https://fal.ai/tmp/img1.png", "width": 1024, "height": 1024},
                {"url": "https://fal.ai/tmp/img2.png", "width": 1024, "height": 1024},
            ]
        }
        mock_fal_client.subscribe.return_value = mock_result

        # Mock httpx for image downloads
        with patch("generationengine.providers.fal_provider.httpx.AsyncClient") as mock_httpx:
            mock_get = AsyncMock()
            mock_get.return_value.content = b"image_data"
            mock_get.return_value.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.get = mock_get

            provider = FalProvider(api_key="test-key")
            result = await provider.generate(
                prompt="A red dragon",
                model="flux-pro",
                num_images=2,
                size=(1024, 1024),
            )

            assert len(result) == 2
            assert all(isinstance(img, bytes) for img in result)
            mock_fal_client.subscribe.assert_called_once()


@pytest.mark.asyncio
async def test_fal_provider_generate_timeout():
    """Test that Fal provider raises RetryableError on timeout."""
    with patch("generationengine.providers.fal_provider.fal_client") as mock_fal_client:
        mock_fal_client.subscribe.side_effect = TimeoutError("Request timed out")

        provider = FalProvider(api_key="test-key")

        from generationengine.services.retry_service import RetryableError
        from generationengine.models.errors import ErrorCode
        with pytest.raises(RetryableError) as exc_info:
            await provider.generate(
                prompt="A red dragon",
                model="flux-pro",
                num_images=1,
                size=(1024, 1024),
            )

        # Verify it's a retryable error with timeout code
        assert exc_info.value.error_code == ErrorCode.PROVIDER_TIMEOUT
        # Message should mention timeout (message says "timed out")
        assert exc_info.value.message and ("timeout" in exc_info.value.message.lower() or "timed out" in exc_info.value.message.lower())

