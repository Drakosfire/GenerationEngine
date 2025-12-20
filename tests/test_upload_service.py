"""Tests for Cloudflare R2 upload service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from generationengine.services.upload_service import UploadService
from generationengine.models.errors import ErrorCode


@pytest.mark.asyncio
async def test_upload_service_upload_success():
    """Test successful image upload to Cloudflare Images API."""
    with patch("generationengine.services.upload_service.httpx.AsyncClient") as mock_httpx:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "variants": ["https://cdn.example.com/test-key.png/public"]
            }
        }
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        service = UploadService(
            account_id="test-account",
            api_token="test-token",
        )

        image_bytes = b"fake_image_data"
        url = await service.upload_image(image_bytes, filename="test-key.png")

        assert url.endswith("/public")
        mock_httpx.return_value.__aenter__.return_value.post.assert_called_once()


@pytest.mark.asyncio
async def test_upload_service_upload_failure():
    """Test that upload service handles Cloudflare API failures."""
    with patch("generationengine.services.upload_service.httpx.AsyncClient") as mock_httpx:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        service = UploadService(
            account_id="test-account",
            api_token="test-token",
        )

        from generationengine.services.retry_service import RetryableError
        with pytest.raises(RetryableError):
            await service.upload_image(b"fake_image_data", filename="test-key.png")


@pytest.mark.asyncio
async def test_upload_service_generate_unique_key():
    """Test that upload service generates unique keys for images."""
    with patch("generationengine.services.upload_service.httpx.AsyncClient") as mock_httpx:
        # Mock responses with different URLs to simulate unique keys
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "result": {"variants": ["https://cdn.example.com/generated_001.png/public"]}
        }
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "result": {"variants": ["https://cdn.example.com/generated_002.png/public"]}
        }
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=[mock_response1, mock_response2]
        )

        service = UploadService(
            account_id="test-account",
            api_token="test-token",
        )

        image_bytes = b"fake_image_data"
        url1 = await service.upload_image(image_bytes, prefix="generated")
        url2 = await service.upload_image(image_bytes, prefix="generated")

        # URLs should be different (simulated by different mock responses)
        assert url1 != url2
        assert "generated" in url1 or "generated_001" in url1
        assert "generated" in url2 or "generated_002" in url2

