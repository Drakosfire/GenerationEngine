"""Product-neutral model catalog, capabilities, and inference profiles."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Capability(str, Enum):
    TEXT = "text"
    STRUCTURED_TEXT = "structured_text"
    STREAMING_TEXT = "streaming_text"
    IMAGE = "image"
    IMAGE_EDIT = "image_edit"


class InferenceProfile(str, Enum):
    TEXT_FAST = "text_fast"
    STRUCTURED_LOW_COST = "structured_low_cost"
    STRUCTURED_HIGH_RELIABILITY = "structured_high_reliability"
    IMAGE_HIGH_QUALITY = "image_high_quality"


class Availability(str, Enum):
    AVAILABLE = "available"
    DEPRECATED = "deprecated"
    UNKNOWN = "unknown"


FORBIDDEN_PROFILE_VOCABULARY = frozenset(
    {
        "statblock",
        "card",
        "map",
        "ruleslawyer",
        "agent",
        "buddy",
        "campaign",
        "runbook",
        "character",
        "store",
    }
)

ACCEPTED_PROFILES: tuple[InferenceProfile, ...] = tuple(InferenceProfile)
ACCEPTED_CAPABILITIES: tuple[Capability, ...] = tuple(Capability)


class PricingDimension(BaseModel):
    """One pricing axis. usd_per_unit is None when the catalog does not know it."""

    name: str = Field(..., min_length=1)
    unit: str = Field(..., min_length=1)
    usd_per_unit: float | None = Field(default=None, ge=0.0)


class ModelRecord(BaseModel):
    provider: str = Field(..., min_length=1)
    provider_model_id: str = Field(..., min_length=1)
    capabilities: tuple[Capability, ...] = ()
    availability: Availability = Availability.UNKNOWN
    pricing: tuple[PricingDimension, ...] = ()
    pricing_source: str | None = None


def profile_contains_product_vocabulary(name: str) -> bool:
    lowered = name.lower()
    return any(stem in lowered for stem in FORBIDDEN_PROFILE_VOCABULARY)
