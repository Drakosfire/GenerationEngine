"""Public client to fal SDK wire controls; no paid requests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from generationengine import (
    FailureCode,
    GenerationClient,
    GenerationEngineError,
    ImageOptions,
    ImageRequest,
)
from generationengine.providers.fal_provider import FalProvider


@pytest.mark.parametrize(
    ("model", "options", "wire"),
    [
        (
            "gpt-image-2.5-flare",
            ImageOptions(quality="medium", output_format="webp"),
            {
                "quality": "medium",
                "output_format": "webp",
                "image_size": {"width": 1024, "height": 1024},
            },
        ),
        (
            "gpt-image-2.5-sunburst",
            ImageOptions(background="transparent"),
            {"background": "transparent", "image_size": {"width": 1024, "height": 1024}},
        ),
        (
            "nano-banana-2.1",
            ImageOptions(seed=0, output_format="jpeg"),
            {"aspect_ratio": "1:1", "resolution": "1K", "seed": 0, "output_format": "jpeg"},
        ),
        (
            "gpt-image-1.5",
            ImageOptions(quality="low", background="transparent", output_format="webp"),
            {"quality": "low", "background": "transparent", "output_format": "webp"},
        ),
        (
            "flux-2-pro",
            ImageOptions(seed=0, output_format="png"),
            {"seed": 0, "output_format": "png"},
        ),
        (
            "gpt-image-1.5",
            None,
            {"quality": "high", "background": "opaque", "output_format": "png"},
        ),
    ],
)
async def test_controls_reach_paid_boundary(model, options, wire):
    with (
        patch("generationengine.providers.fal_provider.fal_client") as sdk,
        patch("generationengine.providers.fal_provider.httpx.AsyncClient") as http,
    ):
        sdk.subscribe_async = AsyncMock(
            return_value={"images": [{"url": "https://fal.media/test.png"}]}
        )
        response = MagicMock(content=b"\x89PNG\r\n\x1a\nimage")
        http.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)
        client = GenerationClient(image_provider=FalProvider(api_key="synthetic"))
        result = await client.generate_image(
            ImageRequest(prompt="A swamp town", model=model, options=options)
        )
        endpoint = {
            "gpt-image-2.5-flare": "openai/gpt-image-2.5/flare/text-to-image",
            "gpt-image-2.5-sunburst": "openai/gpt-image-2.5/sunburst/text-to-image",
            "nano-banana-2.1": "google/nano-banana-2.1",
        }.get(model)
        if endpoint:
            assert sdk.subscribe_async.call_args.args[0] == endpoint
        args = sdk.subscribe_async.call_args.kwargs["arguments"]
        for key, value in wire.items():
            assert args[key] == value
        assert sdk.subscribe_async.call_count == 1
        assert result.images[0].media_type == "image/png"
        assert result.images[0].width is None and result.images[0].height is None


@pytest.mark.parametrize(
    ("model", "options", "size"),
    [
        ("nano-banana-2.1", ImageOptions(quality="high"), (1024, 1024)),
        ("gpt-image-2.5-flare", ImageOptions(seed=9), (1024, 1024)),
        ("gpt-image-2.5-sunburst", ImageOptions(), (777, 777)),
        ("gpt-image-1.5", ImageOptions(seed=9), (1024, 1024)),
        ("flux-2-pro", ImageOptions(quality="low"), (1024, 1024)),
        ("flux-2-pro", ImageOptions(output_format="webp"), (1024, 1024)),
        ("nano-banana-pro", ImageOptions(background="transparent"), (1024, 1024)),
        ("gpt-image-1.5", ImageOptions(quality="high"), (777, 777)),
    ],
)
async def test_unsupported_controls_fail_before_submission(model, options, size):
    with patch("generationengine.providers.fal_provider.fal_client") as sdk:
        sdk.subscribe_async = AsyncMock()
        client = GenerationClient(image_provider=FalProvider(api_key="synthetic"))
        with pytest.raises(GenerationEngineError) as error:
            await client.generate_image(
                ImageRequest(
                    prompt="A map", model=model, options=options, width=size[0], height=size[1]
                )
            )
        assert error.value.failure.code is FailureCode.INVALID_REQUEST
        sdk.subscribe_async.assert_not_called()


@pytest.mark.parametrize(
    ("blob", "mime"),
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xffimage", "image/jpeg"),
        (b"RIFF0000WEBPimage", "image/webp"),
        (b"unknown", "application/octet-stream"),
    ],
)
async def test_output_encoding_is_not_guessed_from_request(blob, mime):
    class Provider:
        async def generate(self, **kwargs):
            return [blob]

    result = await GenerationClient(image_provider=Provider()).generate_image(
        ImageRequest(prompt="map", model="gpt-image-1.5")
    )
    assert result.images[0].media_type == mime
    assert result.images[0].width is None and result.images[0].height is None


def test_options_reject_unknown_or_invalid_parameters():
    for args in [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"quality": "ultra"},
        {"output_format": "svg"},
        {"made_up": 1},
    ]:
        with pytest.raises(ValidationError):
            ImageOptions(**args)


async def test_nano_ultrawide_uses_provider_ratio_spelling():
    with (
        patch("generationengine.providers.fal_provider.fal_client") as sdk,
        patch("generationengine.providers.fal_provider.httpx.AsyncClient") as http,
    ):
        sdk.subscribe_async = AsyncMock(
            return_value={"images": [{"url": "https://fal.media/test.png"}]}
        )
        response = MagicMock(content=b"\x89PNG\r\n\x1a\nimage")
        http.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)
        client = GenerationClient(image_provider=FalProvider(api_key="synthetic"))
        await client.generate_image(ImageRequest(
            prompt="An ultrawide swamp", model="nano-banana-2.1", width=2100, height=900,
        ))
        assert sdk.subscribe_async.call_count == 1
        assert sdk.subscribe_async.call_args.args == ("google/nano-banana-2.1",)
        assert sdk.subscribe_async.call_args.kwargs["arguments"]["aspect_ratio"] == "21:9"
