"""Tests for Fal image provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from generationengine.providers.fal_provider import FalProvider


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
        mock_fal_client.subscribe_async = AsyncMock(return_value=mock_result)

        # Mock httpx for image downloads
        with patch("generationengine.providers.fal_provider.httpx.AsyncClient") as mock_httpx:
            mock_get = AsyncMock()
            mock_get.return_value.content = b"image_data"
            mock_get.return_value.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.get = mock_get

            provider = FalProvider(api_key="test-key")
            result = await provider.generate(
                prompt="A red dragon",
                model="flux-2-pro",
                num_images=2,
                size=(1024, 1024),
            )

            assert len(result) == 2
            assert all(isinstance(img, bytes) for img in result)
            mock_fal_client.subscribe_async.assert_called_once()


@pytest.mark.asyncio
async def test_fal_provider_generate_timeout():
    """Test that Fal provider raises RetryableError on timeout."""
    with patch("generationengine.providers.fal_provider.fal_client") as mock_fal_client:
        mock_fal_client.subscribe_async = AsyncMock(side_effect=TimeoutError("Request timed out"))

        provider = FalProvider(api_key="test-key")

        from generationengine.failures import FailureCode
        from generationengine.providers.errors import ProviderError
        with pytest.raises(ProviderError) as exc_info:
            await provider.generate(
                prompt="A red dragon",
                model="flux-2-pro",
                num_images=1,
                size=(1024, 1024),
            )

        assert exc_info.value.failure.code is FailureCode.PROVIDER_TIMEOUT
        assert "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_fal_provider_inpainting_flux_2_pro():
    """Test successful inpainting with FLUX 2 Pro via /edit endpoint."""
    with patch("generationengine.providers.fal_provider.fal_client") as mock_fal_client:
        # Mock Fal response
        mock_result = {
            "images": [
                {"url": "https://fal.ai/tmp/img1.png", "width": 1024, "height": 1024},
            ]
        }
        mock_fal_client.subscribe_async = AsyncMock(return_value=mock_result)

        # Mock httpx for image downloads
        with patch("generationengine.providers.fal_provider.httpx.AsyncClient") as mock_httpx:
            mock_get = AsyncMock()
            mock_get.return_value.content = b"image_data"
            mock_get.return_value.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.get = mock_get

            provider = FalProvider(api_key="test-key")
            result = await provider.generate(
                prompt="Add a dragon",
                model="flux-2-pro",
                num_images=1,
                size=(1024, 1024),
                base_image_base64="data:image/png;base64,abc123",
            )

            assert len(result) == 1
            assert all(isinstance(img, bytes) for img in result)

            # Verify fal_client.subscribe_async was called with inpainting endpoint
            mock_fal_client.subscribe_async.assert_called_once()
            call_args = mock_fal_client.subscribe_async.call_args
            assert call_args[0][0] == "fal-ai/flux-2-pro/edit"
            assert call_args[1]["arguments"]["image_urls"] == ["data:image/png;base64,abc123"]


@pytest.mark.asyncio
async def test_fal_provider_inpainting_gpt_image_15():
    """Test inpainting with GPT Image 1.5 via /edit endpoint with mask."""
    with patch("generationengine.providers.fal_provider.fal_client") as mock_fal_client:
        mock_result = {"images": [{"url": "https://fal.ai/tmp/img1.png"}]}
        mock_fal_client.subscribe_async = AsyncMock(return_value=mock_result)

        with patch("generationengine.providers.fal_provider.httpx.AsyncClient") as mock_httpx:
            mock_get = AsyncMock()
            mock_get.return_value.content = b"image_data"
            mock_get.return_value.raise_for_status = MagicMock()
            mock_httpx.return_value.__aenter__.return_value.get = mock_get

            provider = FalProvider(api_key="test-key")
            await provider.generate(
                prompt="Add fire effects",
                model="gpt-image-1.5",
                num_images=1,
                size=(1024, 1024),
                base_image_base64="data:image/png;base64,base",
                mask_base64="data:image/png;base64,mask",
            )

            # Verify correct endpoint and mask parameter
            call_args = mock_fal_client.subscribe_async.call_args
            assert call_args[0][0] == "fal-ai/gpt-image-1.5/edit"
            assert call_args[1]["arguments"]["image_urls"] == ["data:image/png;base64,base"]
            assert call_args[1]["arguments"]["mask_image_url"] == "data:image/png;base64,mask"



@pytest.mark.asyncio
@pytest.mark.parametrize("editing", [False, True])
async def test_real_fal_sdk_content_refusal_is_safe_and_never_retried(editing):
    import httpx
    from fal_client.client import _raise_for_status

    from generationengine import GenerationClient, GenerationEngineError, ImageRequest
    from generationengine.failures import FailureCode, Retryability

    response = httpx.Response(422, request=httpx.Request("GET", "https://queue.fal.run/test/model/requests/test"),
        json={"detail": [{"loc": ["body", "prompt"], "type": "content_policy_violation",
                         "msg": "PRIVATE_PROVIDER_DETAIL", "input": "PRIVATE_PROMPT"}]})
    with pytest.raises(Exception) as captured:
        _raise_for_status(response)
    sdk_error = captured.value
    assert isinstance(sdk_error.__cause__, httpx.HTTPStatusError)
    with patch("generationengine.providers.fal_provider.fal_client.subscribe_async",
               new=AsyncMock(side_effect=sdk_error)) as submit:
        client = GenerationClient(image_provider=FalProvider(api_key="synthetic-not-secret"))
        request = ImageRequest(prompt="Synthetic fantasy creature", model="flux-2-pro",
            **({"base_image_base64": "data:image/png;base64,synthetic"} if editing else {}))
        try:
            with pytest.raises(GenerationEngineError) as failure:
                await (client.edit_image(request) if editing else client.generate_image(request))
        finally:
            await client.aclose()
        error = failure.value
        assert error.failure.code is FailureCode.PROVIDER_REFUSED
        assert error.failure.retryability is Retryability.NO
        assert error.observation.state.value == "refused"
        assert error.observation.retry_count == 0
        assert submit.await_count == 1
        assert "PRIVATE_" not in str(error) and "PRIVATE_" not in error.failure.model_dump_json()
        assert error.__cause__.__cause__ is sdk_error


@pytest.mark.asyncio
@pytest.mark.parametrize("status,payload", [
    (422, {"detail": [{"type": "value_error", "msg": "content_policy_violation"}]}),
    (500, {"detail": [{"type": "content_policy_violation"}]}),
    (422, {"detail": "content_policy_violation"}),
    (422, {"detail": [None, {"msg": "PRIVATE_DETAIL"}]}),
    (422, ["content_policy_violation"]),
    (422, None),
])
async def test_fal_sdk_unknown_errors_are_not_guessed_to_be_refusals(status, payload):
    import httpx
    from fal_client.client import _raise_for_status

    from generationengine.failures import FailureCode
    from generationengine.providers.errors import ProviderError

    response = httpx.Response(status, request=httpx.Request("GET", "https://queue.fal.run/test/model/requests/test"),
        **({"content": b"not JSON"} if payload is None else {"json": payload}))
    with pytest.raises(Exception) as captured:
        _raise_for_status(response)
    with patch("generationengine.providers.fal_provider.fal_client.subscribe_async",
               new=AsyncMock(side_effect=captured.value)) as submit:
        with pytest.raises(ProviderError) as failure:
            await FalProvider(api_key="synthetic-not-secret").generate(
                prompt="Synthetic fantasy creature", model="flux-2-pro", size=(1024, 1024), num_images=1)
        assert failure.value.failure.code is FailureCode.PROVIDER_ERROR
        assert failure.value.failure.message == "Provider request failed."
        assert submit.await_count == 1
