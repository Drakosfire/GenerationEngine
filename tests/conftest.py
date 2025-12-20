"""Shared pytest fixtures for GenerationEngine tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from generationengine.providers.base import ImageProvider


class MockImageProvider:
    """Mock image provider for testing."""

    def __init__(self, should_fail: bool = False, fail_count: int = 0):
        """
        Initialize mock provider.

        Args:
            should_fail: If True, provider will raise exceptions
            fail_count: Number of times to fail before succeeding (for retry tests)
        """
        self.should_fail = should_fail
        self.fail_count = fail_count
        self.call_count = 0

    async def generate(
        self, prompt: str, model: str, num_images: int, size: tuple[int, int]
    ) -> list[bytes]:
        """Mock generate method."""
        self.call_count += 1

        if self.should_fail and self.call_count <= self.fail_count:
            raise Exception(f"Mock provider failure (call {self.call_count})")

        # Return mock image bytes
        return [b"mock_image_data"] * num_images


@pytest.fixture
def mock_image_provider():
    """Fixture for a working mock image provider."""
    return MockImageProvider(should_fail=False)


@pytest.fixture
def failing_image_provider():
    """Fixture for a failing mock image provider."""
    return MockImageProvider(should_fail=True, fail_count=999)


@pytest.fixture
def retryable_image_provider():
    """Fixture for a provider that fails then succeeds (for retry tests)."""
    return MockImageProvider(should_fail=True, fail_count=2)


@pytest.fixture
def mock_firestore():
    """Fixture for mock Firestore database."""
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection
    return mock_db, mock_collection


@pytest.fixture
def mock_cloudflare_r2():
    """Fixture for mock Cloudflare R2 storage."""
    mock_r2 = AsyncMock()
    mock_r2.upload.return_value = "https://r2.example.com/image.png"
    return mock_r2
