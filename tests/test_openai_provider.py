"""Tests for OpenAI image provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from generationengine.providers.openai_provider import OpenAIImageProvider


@pytest.mark.asyncio
async def test_openai_provider_generate_success():
    """Test successful image generation with OpenAI provider."""
    with patch("generationengine.providers.openai_provider.OpenAI") as mock_openai_class:
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock OpenAI response - generate is synchronous but run in executor
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(url="https://oaidalleapiprodscus.blob.core.windows.net/img1.png", b64_json="base64data1"),
            MagicMock(url="https://oaidalleapiprodscus.blob.core.windows.net/img2.png", b64_json="base64data2"),
        ]
        # generate is synchronous, returns the response directly
        mock_client.images.generate = MagicMock(return_value=mock_response)

        # Since we're using b64_json in the mock, we need to mock base64 decode
        import base64
        with patch("generationengine.providers.openai_provider.base64.b64decode") as mock_b64decode:
            mock_b64decode.side_effect = [b"image_data_1", b"image_data_2"]
            
            provider = OpenAIImageProvider(api_key="test-key")
            result = await provider.generate(
                prompt="A red dragon",
                model="openai",
                num_images=2,
                size=(1024, 1024),
            )

            assert len(result) == 2
            assert all(isinstance(img, bytes) for img in result)
            assert result[0] == b"image_data_1"
            assert result[1] == b"image_data_2"


@pytest.mark.asyncio
async def test_openai_provider_generate_rate_limit():
    """Test that OpenAI provider handles rate limit errors."""
    with patch("generationengine.providers.openai_provider.OpenAI") as mock_openai_class:
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock rate limit error - create a mock exception since openai may not be installed
        class MockRateLimitError(Exception):
            pass
        
        RateLimitError = MockRateLimitError  # Use mock if openai not available
        try:
            from openai import RateLimitError as RealRateLimitError
            RateLimitError = RealRateLimitError
        except ImportError:
            pass

        mock_client.images.generate = AsyncMock(side_effect=RateLimitError("Rate limit exceeded"))

        provider = OpenAIImageProvider(api_key="test-key")

        with pytest.raises(Exception):  # Should raise appropriate error
            await provider.generate(
                prompt="A red dragon",
                model="openai",
                num_images=1,
                size=(1024, 1024),
            )

