"""Legacy metrics must not retain full prompts. Numeric consumer fields remain."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from generationengine.models.requests import (
    ImageGenerationRequest,
    ImageModel,
    ImageSize,
    TextGenerationRequest,
    TextModel,
)
from generationengine.services.image_service import ImageService
from generationengine.services.text_service import TextGenerationService
from generationengine.telemetry import bounded_image_metrics_input, bounded_text_metrics_input

TEXT_SENTINEL = "UNIQUE_TEXT_PROMPT_SENTINEL_E2B"
IMAGE_SENTINEL = "UNIQUE_IMAGE_PROMPT_SENTINEL_E2B"


def test_bounded_text_metrics_omit_prompt_bodies() -> None:
    request = TextGenerationRequest(
        system_prompt=TEXT_SENTINEL,
        user_prompt=TEXT_SENTINEL,
        model=TextModel.GPT_5_1,
        response_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
        response_schema_name="NeutralCount",
    )
    payload = bounded_text_metrics_input(request)
    serialized = json.dumps(payload)
    assert TEXT_SENTINEL not in serialized
    assert payload["system_prompt_length"] == len(TEXT_SENTINEL)
    assert payload["user_prompt_length"] == len(TEXT_SENTINEL)
    assert payload["schema_name"] == "NeutralCount"
    assert "schema_hash" in payload


def test_bounded_image_metrics_omit_prompt_bodies() -> None:
    request = ImageGenerationRequest(
        prompt=IMAGE_SENTINEL,
        model=ImageModel.FLUX_2_PRO,
        num_images=2,
        size=ImageSize.SQUARE,
        negative_prompt=IMAGE_SENTINEL,
    )
    payload = bounded_image_metrics_input(request)
    serialized = json.dumps(payload)
    assert IMAGE_SENTINEL not in serialized
    assert payload["image_prompt_length"] == len(IMAGE_SENTINEL)
    assert payload["num_images"] == 2
    assert payload["has_negative_prompt"] is True


@pytest.mark.asyncio
async def test_text_service_metrics_omit_sentinel_and_keep_numeric_fields() -> None:
    response_obj = MagicMock()
    response_obj.output_text = "ok"
    response_obj.refusal = None
    response_obj.usage = MagicMock()
    response_obj.usage.input_tokens = 11
    response_obj.usage.output_tokens = 7
    response_obj.usage.total_tokens = 18

    with patch("generationengine.services.text_service.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        service = TextGenerationService(openai_api_key="test-key")
        service.openai_client = mock_client
        mock_client.responses.create = AsyncMock(return_value=response_obj)

        result = await service.generate(
            TextGenerationRequest(system_prompt=TEXT_SENTINEL, user_prompt=TEXT_SENTINEL)
        )

    assert result.success is True
    assert result.metrics is not None
    assert result.metrics.tokens_used == 18
    assert result.metrics.model_used == "gpt-5.1"
    assert result.metrics.duration_ms >= 0
    assert TEXT_SENTINEL not in (result.metrics.input or "")
    assert TEXT_SENTINEL not in (result.metrics.output or "")


@pytest.mark.asyncio
async def test_image_service_metrics_omit_sentinel() -> None:
    class _Provider:
        async def generate(self, *args, **kwargs):
            return [b"png-bytes"]

    class _Uploader:
        async def upload_image(self, image_bytes, prefix="generated", filename=None):
            return "https://example.invalid/x.png"

    service = ImageService(upload_service=_Uploader())
    service.providers = {"flux-2-pro": _Provider()}
    result = await service.generate(
        ImageGenerationRequest(prompt=IMAGE_SENTINEL, model=ImageModel.FLUX_2_PRO, num_images=1)
    )
    assert result.success is True
    assert result.metrics is not None
    assert result.metrics.model_used == "flux-2-pro"
    assert result.metrics.retry_count == 0
    assert result.metrics.duration_ms >= 0
    assert IMAGE_SENTINEL not in (result.metrics.input or "")
    assert IMAGE_SENTINEL not in (result.metrics.output or "")
