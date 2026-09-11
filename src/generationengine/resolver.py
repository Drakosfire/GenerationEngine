"""Live catalog and profile/model resolution.

Populated only with models required by the paired DungeonMindServer cutover.
Product action names never appear here.
"""

from __future__ import annotations

from generationengine.catalog import (
    Availability,
    Capability,
    InferenceProfile,
    ModelRecord,
)
from generationengine.failures import FailureCode, InferenceFailure


class ResolutionError(Exception):
    def __init__(self, failure: InferenceFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure

# Catalog selection keys match current product aliases / provider-native IDs.
LIVE_MODELS: dict[str, ModelRecord] = {
    "gpt-5.1": ModelRecord(
        provider="openai",
        provider_model_id="gpt-5.1",
        capabilities=(
            Capability.TEXT,
            Capability.STRUCTURED_TEXT,
            Capability.STREAMING_TEXT,
        ),
        availability=Availability.AVAILABLE,
    ),
    "gpt-4o": ModelRecord(
        provider="openai",
        provider_model_id="gpt-4o",
        capabilities=(
            Capability.TEXT,
            Capability.STRUCTURED_TEXT,
            Capability.STREAMING_TEXT,
        ),
        availability=Availability.AVAILABLE,
    ),
    "gpt-5.6-luna": ModelRecord(
        provider="openai",
        provider_model_id="gpt-5.6-luna",
        capabilities=(Capability.STRUCTURED_TEXT, Capability.TEXT),
        availability=Availability.AVAILABLE,
    ),
    "flux-2-pro": ModelRecord(
        provider="fal",
        provider_model_id="flux-2-pro",
        capabilities=(Capability.IMAGE, Capability.IMAGE_EDIT),
        availability=Availability.AVAILABLE,
    ),
    "nano-banana-pro": ModelRecord(
        provider="fal",
        provider_model_id="nano-banana-pro",
        capabilities=(Capability.IMAGE, Capability.IMAGE_EDIT),
        availability=Availability.AVAILABLE,
    ),
    "gpt-image-1.5": ModelRecord(
        provider="fal",
        provider_model_id="gpt-image-1.5",
        capabilities=(Capability.IMAGE, Capability.IMAGE_EDIT),
        availability=Availability.AVAILABLE,
    ),
    "flux-lora-i2i": ModelRecord(
        provider="fal",
        provider_model_id="flux-lora-i2i",
        capabilities=(Capability.IMAGE,),
        availability=Availability.AVAILABLE,
    ),
}

PROFILE_DEFAULTS: dict[InferenceProfile, str] = {
    InferenceProfile.TEXT_FAST: "gpt-5.1",
    InferenceProfile.STRUCTURED_LOW_COST: "gpt-5.1",
    InferenceProfile.STRUCTURED_HIGH_RELIABILITY: "gpt-5.6-luna",
    InferenceProfile.IMAGE_HIGH_QUALITY: "flux-2-pro",
    InferenceProfile.IMAGE_EDIT_HIGH_QUALITY: "gpt-image-1.5",
}


class Resolution:
    def __init__(
        self,
        *,
        catalog_id: str,
        record: ModelRecord,
        profile: InferenceProfile | None,
        requested_model: str | None,
    ) -> None:
        self.catalog_id = catalog_id
        self.record = record
        self.profile = profile
        self.requested_model = requested_model


def resolve(
    *,
    capability: Capability,
    profile: InferenceProfile | None = None,
    model: str | None = None,
) -> Resolution:
    """Resolve a profile and/or explicit catalog model to a live record.

    Explicit valid model selection wins over the profile default.
    """
    requested_model = model.strip() if model and model.strip() else None
    if requested_model:
        record = LIVE_MODELS.get(requested_model)
        if record is None or capability not in record.capabilities:
            raise ResolutionError(
                InferenceFailure.from_code(
                    FailureCode.UNSUPPORTED_CAPABILITY,
                    f"Model {requested_model!r} does not support {capability.value}.",
                )
            )
        return Resolution(
            catalog_id=requested_model,
            record=record,
            profile=profile,
            requested_model=requested_model,
        )
    if profile is None:
        raise ResolutionError(
            InferenceFailure.from_code(
                FailureCode.INVALID_REQUEST,
                "A generic inference profile or explicit catalog model is required.",
            )
        )
    catalog_id = PROFILE_DEFAULTS.get(profile)
    if catalog_id is None:
        raise ResolutionError(
            InferenceFailure.from_code(
                FailureCode.UNSUPPORTED_CAPABILITY,
                f"Profile {profile.value!r} has no live model mapping.",
            )
        )
    record = LIVE_MODELS[catalog_id]
    if capability not in record.capabilities:
        raise ResolutionError(
            InferenceFailure.from_code(
                FailureCode.UNSUPPORTED_CAPABILITY,
                f"Profile {profile.value!r} does not support {capability.value}.",
            )
        )
    return Resolution(
        catalog_id=catalog_id,
        record=record,
        profile=profile,
        requested_model=None,
    )
