"""Current-install characterization.

These encode baseline packaging facts that successor slices will change.
They are not target invariants for Cloudflare coupling, SSE, or TextModel.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from generationengine.services.image_service import ImageService
from generationengine.services.text_service import TextGenerationService
from generationengine.services.upload_service import UploadService


def _without_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OPENAI_API_KEY",
        "FAL_KEY",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_IMAGES_API_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def test_clean_install_does_not_include_fal_client() -> None:
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.version("fal-client")


def test_text_service_requires_openai_key_today(monkeypatch: pytest.MonkeyPatch) -> None:
    _without_provider_env(monkeypatch)
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        TextGenerationService()


def test_upload_and_image_service_require_cloudflare_today(monkeypatch: pytest.MonkeyPatch) -> None:
    """Current coupling. E2D must make image construction possible without Cloudflare."""
    _without_provider_env(monkeypatch)
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        UploadService()
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        ImageService()


def test_openai_image_provider_exists_but_is_unwired(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advertised OpenAI image path is not registered on ImageService today."""
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_IMAGES_API_TOKEN", "test-token")
    monkeypatch.delenv("FAL_KEY", raising=False)
    service = ImageService()
    assert "openai" not in service.providers
    from generationengine.providers.openai_provider import OpenAIImageProvider

    assert OpenAIImageProvider is not None
    assert all(type(provider).__name__ != "OpenAIImageProvider" for provider in service.providers.values())
