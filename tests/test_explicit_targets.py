"""Explicit provider+model targets are executable without catalog membership."""

from __future__ import annotations

import pytest

from generationengine import (
    Capability,
    FailureCode,
    GenerationClient,
    GenerationEngineError,
    InferenceProfile,
    ObservationState,
    TextGenerationCall,
    TextGenerationResult,
    TextRequest,
)
from generationengine.resolver import LIVE_MODELS, ResolutionError, resolve


class _FakeTextProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.models: list[str] = []

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        self.calls += 1
        self.models.append(call.model)
        return TextGenerationResult(text="ok", response_model=call.model)

    async def stream(self, call: TextGenerationCall):
        raise NotImplementedError


UNCATEGORIZED_OPENAI = "gpt-5.3-codex"
UNCATEGORIZED_OPENROUTER = "deepseek/deepseek-v4.1-flash"


def test_explicit_openai_target_skips_catalog() -> None:
    assert UNCATEGORIZED_OPENAI not in LIVE_MODELS
    resolution = resolve(
        capability=Capability.TEXT,
        provider="openai",
        model=UNCATEGORIZED_OPENAI,
    )
    assert resolution.provider == "openai"
    assert resolution.provider_model_id == UNCATEGORIZED_OPENAI
    assert resolution.catalog_id is None
    assert resolution.record is None
    assert resolution.resolved_model == UNCATEGORIZED_OPENAI
    assert resolution.pricing_source is None


def test_explicit_openrouter_target_skips_catalog() -> None:
    assert UNCATEGORIZED_OPENROUTER not in LIVE_MODELS
    resolution = resolve(
        capability=Capability.STRUCTURED_TEXT,
        profile=InferenceProfile.STRUCTURED_LOW_COST,
        provider="OpenRouter",
        model=UNCATEGORIZED_OPENROUTER,
    )
    assert resolution.provider == "openrouter"
    assert resolution.provider_model_id == UNCATEGORIZED_OPENROUTER
    assert resolution.catalog_id is None
    assert resolution.record is None
    assert resolution.profile is InferenceProfile.STRUCTURED_LOW_COST
    assert resolution.resolved_model == UNCATEGORIZED_OPENROUTER


def test_provider_without_model_is_invalid_request() -> None:
    with pytest.raises(ResolutionError) as exc:
        resolve(capability=Capability.TEXT, provider="openrouter")
    assert exc.value.failure.code is FailureCode.INVALID_REQUEST


def test_unknown_provider_is_unsupported() -> None:
    with pytest.raises(ResolutionError) as exc:
        resolve(capability=Capability.TEXT, provider="unknown-provider", model="x")
    assert exc.value.failure.code is FailureCode.UNSUPPORTED_CAPABILITY


def test_model_only_uncataloged_target_fails_closed() -> None:
    with pytest.raises(ResolutionError) as exc:
        resolve(capability=Capability.TEXT, model=UNCATEGORIZED_OPENROUTER)
    assert exc.value.failure.code is FailureCode.UNSUPPORTED_CAPABILITY
    assert UNCATEGORIZED_OPENROUTER not in LIVE_MODELS


def test_profile_defaults_are_unchanged() -> None:
    assert resolve(capability=Capability.TEXT, profile=InferenceProfile.TEXT_FAST).catalog_id == "gpt-5.1"
    assert (
        resolve(
            capability=Capability.STRUCTURED_TEXT,
            profile=InferenceProfile.STRUCTURED_LOW_COST,
        ).catalog_id
        == "gpt-5.1"
    )
    assert (
        resolve(
            capability=Capability.STRUCTURED_TEXT,
            profile=InferenceProfile.STRUCTURED_HIGH_RELIABILITY,
        ).catalog_id
        == "gpt-5.6-luna"
    )


@pytest.mark.asyncio
async def test_profile_only_still_dispatches_to_openai() -> None:
    openai = _FakeTextProvider()
    openrouter = _FakeTextProvider()
    client = GenerationClient(text_providers={"openai": openai, "openrouter": openrouter})
    result = await client.generate_text(
        TextRequest(user_prompt="hi", profile=InferenceProfile.TEXT_FAST)
    )
    assert openai.calls == 1
    assert openrouter.calls == 0
    assert result.observation.provider == "openai"
    assert result.observation.resolved_model == "gpt-5.1"


@pytest.mark.asyncio
async def test_explicit_target_dispatches_only_to_named_provider() -> None:
    openai = _FakeTextProvider()
    openrouter = _FakeTextProvider()
    client = GenerationClient(text_providers={"openai": openai, "openrouter": openrouter})
    result = await client.generate_text(
        TextRequest(
            user_prompt="hi",
            profile=InferenceProfile.STRUCTURED_LOW_COST,
            provider="openrouter",
            model=UNCATEGORIZED_OPENROUTER,
        )
    )
    assert openrouter.calls == 1
    assert openai.calls == 0
    assert result.observation.provider == "openrouter"
    assert result.observation.requested_model == UNCATEGORIZED_OPENROUTER
    assert result.observation.resolved_model == UNCATEGORIZED_OPENROUTER
    assert result.observation.requested_profile == "structured_low_cost"
    assert result.observation.cost_usd is None
    assert result.observation.pricing_source is None
    assert result.observation.state is ObservationState.COMPLETED


@pytest.mark.asyncio
async def test_uncataloged_explicit_model_executes_while_catalog_mode_fails() -> None:
    openai = _FakeTextProvider()
    client = GenerationClient(text_provider=openai)
    result = await client.generate_text(
        TextRequest(user_prompt="hi", provider="openai", model=UNCATEGORIZED_OPENAI)
    )
    assert openai.calls == 1
    assert result.observation.provider == "openai"
    assert result.observation.resolved_model == UNCATEGORIZED_OPENAI
    assert UNCATEGORIZED_OPENAI not in LIVE_MODELS

    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_text(TextRequest(user_prompt="hi", model=UNCATEGORIZED_OPENAI))
    assert exc.value.failure.code is FailureCode.UNSUPPORTED_CAPABILITY
    assert openai.calls == 1
