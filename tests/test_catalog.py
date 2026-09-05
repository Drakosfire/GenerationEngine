"""Catalog and inference-profile authority tests."""

from __future__ import annotations

from generationengine.catalog import (
    ACCEPTED_CAPABILITIES,
    ACCEPTED_PROFILES,
    Availability,
    Capability,
    InferenceProfile,
    ModelRecord,
    PricingDimension,
    profile_contains_product_vocabulary,
)


def test_capability_vocabulary_is_product_neutral() -> None:
    assert {item.value for item in ACCEPTED_CAPABILITIES} == {
        "text",
        "structured_text",
        "streaming_text",
        "image",
        "image_edit",
    }


def test_accepted_profiles_exist_and_are_requirement_shaped() -> None:
    assert {item.value for item in ACCEPTED_PROFILES} == {
        "text_fast",
        "structured_low_cost",
        "structured_high_reliability",
        "image_high_quality",
    }
    for profile in InferenceProfile:
        assert not profile_contains_product_vocabulary(profile.value)


def test_forbidden_product_vocabulary_is_rejected() -> None:
    for name in (
        "statblock_generation",
        "card_image",
        "map_prompt_compilation",
        "ruleslawyer_response",
        "agent_turn",
        "buddy_planner",
        "campaign_summary",
        "runbook_step",
    ):
        assert profile_contains_product_vocabulary(name)


def test_unknown_pricing_and_availability_remain_representable() -> None:
    record = ModelRecord(
        provider="openai",
        provider_model_id="gpt-5.1",
        capabilities=(Capability.TEXT, Capability.STRUCTURED_TEXT),
        availability=Availability.UNKNOWN,
        pricing=(),
        pricing_source=None,
    )
    assert record.pricing == ()
    assert record.pricing_source is None
    assert record.availability is Availability.UNKNOWN

    unknown_price = PricingDimension(name="input_tokens", unit="token", usd_per_unit=None)
    assert unknown_price.usd_per_unit is None

    image_price = PricingDimension(name="image", unit="image", usd_per_unit=0.04)
    assert image_price.usd_per_unit == 0.04
