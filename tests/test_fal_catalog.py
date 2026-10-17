"""Catalog discovery through the public client; no paid inference calls."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from generationengine import FailureCode, GenerationClient, GenerationEngineError
from generationengine.providers.fal_provider import FalProvider


async def test_catalog_page_preserves_discovery_and_schema_errors():
    payload = {
        "models": [
            {
                "endpoint_id": "openai/gpt-image-2.5/flare/text-to-image",
                "metadata": {"display_name": "Flare", "is_favorited": True, "updated_at": "today"},
                "openapi": {"error": {"code": "expansion_failed"}},
            },
            {"endpoint_id": "future/new-model", "metadata": {"display_name": "Future"}},
        ],
        "has_more": True,
        "next_cursor": "Mg",
    }
    with (
        patch("generationengine.providers.fal_provider.fal_client") as sdk,
        patch("generationengine.providers.fal_provider.httpx.AsyncClient") as http,
    ):
        get = AsyncMock(
            return_value=httpx.Response(
                200, json=payload, request=httpx.Request("GET", "https://api.fal.ai/v1/models")
            )
        )
        http.return_value.__aenter__.return_value.get = get
        page = await GenerationClient(
            image_provider=FalProvider(api_key="synthetic")
        ).list_image_models(
            cursor="MQ",
            limit=10,
            include_schema=True,
            endpoint_id="openai/gpt-image-2.5/flare/text-to-image",
        )
        assert page.provider == "fal" and page.next_cursor == "Mg" and page.has_more
        assert page.fetched_at.tzinfo is not None
        assert page.models[0].executable_model == "gpt-image-2.5-flare"
        assert page.models[1].executable_model is None
        assert "is_favorited" not in page.models[0].metadata
        assert page.models[0].openapi["error"]["code"] == "expansion_failed"
        assert get.call_args.args == ("https://api.fal.ai/v1/models",)
        assert get.call_args.kwargs["params"] == {
            "category": "text-to-image",
            "status": "active",
            "limit": "10",
            "cursor": "MQ",
            "expand": "openapi-3.0",
            "endpoint_id": "openai/gpt-image-2.5/flare/text-to-image",
        }
        assert get.call_args.kwargs["headers"] == {"Authorization": "Key synthetic"}
        assert http.call_args.kwargs == {"timeout": 15.0, "follow_redirects": False}
        sdk.subscribe_async.assert_not_called()


@pytest.mark.parametrize(
    ("status", "payload", "code"),
    [
        (401, {"error": "SECRET"}, FailureCode.CONFIGURATION_UNAVAILABLE),
        (403, {"error": "SECRET"}, FailureCode.CONFIGURATION_UNAVAILABLE),
        (429, {"error": "SECRET"}, FailureCode.RATE_LIMITED),
        (500, {"error": "SECRET"}, FailureCode.PROVIDER_UNAVAILABLE),
        (200, {"models": [], "has_more": True}, FailureCode.MALFORMED_PROVIDER_RESPONSE),
        (
            200,
            {"models": [{"endpoint_id": ""}], "has_more": False},
            FailureCode.MALFORMED_PROVIDER_RESPONSE,
        ),
    ],
)
async def test_catalog_failures_are_public_and_sanitized(status, payload, code):
    with (
        patch("generationengine.providers.fal_provider.fal_client"),
        patch("generationengine.providers.fal_provider.httpx.AsyncClient") as http,
    ):
        get = AsyncMock(
            return_value=httpx.Response(
                status, json=payload, request=httpx.Request("GET", "https://api.fal.ai/v1/models")
            )
        )
        http.return_value.__aenter__.return_value.get = get
        with pytest.raises(GenerationEngineError) as error:
            await GenerationClient(
                image_provider=FalProvider(api_key="synthetic")
            ).list_image_models()
        assert error.value.failure.code is code
        assert "SECRET" not in str(error.value)
        assert get.call_count == 1


async def test_catalog_rejects_query_before_http_and_closed_client():
    with (
        patch("generationengine.providers.fal_provider.fal_client"),
        patch("generationengine.providers.fal_provider.httpx.AsyncClient") as http,
    ):
        client = GenerationClient(image_provider=FalProvider(api_key="synthetic"))
        with pytest.raises(GenerationEngineError) as error:
            await client.list_image_models(limit=101)
        assert error.value.failure.code is FailureCode.INVALID_REQUEST
        http.assert_not_called()
        await client.aclose()
        with pytest.raises(GenerationEngineError):
            await client.list_image_models()
