"""Live catalog and profile/model resolution.

Populated only with models required by the paired DungeonMindServer cutover.
Product action names never appear here.

`LIVE_MODELS` is selection and metadata authority: models GenerationEngine may
choose automatically through generic profiles, or describe authoritatively.
It is not an execution allowlist. Explicit provider+model targets may execute
through a registered provider without a catalog row.
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

REGISTERED_TEXT_PROVIDERS: frozenset[str] = frozenset({"openai", "openrouter"})


class Resolution:
    def __init__(
        self,
        *,
        catalog_id: str | None,
        record: ModelRecord | None,
        profile: InferenceProfile | None,
        requested_model: str | None,
        provider: str | None = None,
        provider_model_id: str | None = None,
    ) -> None:
        if provider is None:
            if record is None:
                raise ValueError("Resolution requires provider or a catalog record")
            provider = record.provider
        if provider_model_id is None:
            if record is None:
                raise ValueError("Resolution requires provider_model_id or a catalog record")
            provider_model_id = record.provider_model_id
        self.catalog_id = catalog_id
        self.record = record
        self.profile = profile
        self.requested_model = requested_model
        self.provider = provider
        self.provider_model_id = provider_model_id

    @property
    def resolved_model(self) -> str:
        if self.catalog_id is not None:
            return self.catalog_id
        return self.provider_model_id

    @property
    def pricing_source(self) -> str | None:
        return self.record.pricing_source if self.record is not None else None


def _stripped(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def resolve(
    *,
    capability: Capability,
    profile: InferenceProfile | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Resolution:
    """Resolve a profile, catalog model, or explicit provider+model target.

    Precedence:

    1. explicit provider + explicit model — caller-selected; catalog membership
       is not required
    2. explicit model without provider — strict catalog model resolution
    3. profile without explicit target — catalog/profile resolution
    4. provider without model — INVALID_REQUEST
    5. neither profile nor model — INVALID_REQUEST
    """
    requested_provider = _stripped(provider)
    requested_model = _stripped(model)

    if requested_provider and not requested_model:
        raise ResolutionError(
            InferenceFailure.from_code(
                FailureCode.INVALID_REQUEST,
                "Explicit provider selection requires an explicit model.",
            )
        )

    if requested_provider and requested_model:
        registered = requested_provider.lower()
        if registered not in REGISTERED_TEXT_PROVIDERS:
            raise ResolutionError(
                InferenceFailure.from_code(
                    FailureCode.UNSUPPORTED_CAPABILITY,
                    f"Text provider {registered!r} is not registered.",
                )
            )
        return Resolution(
            catalog_id=None,
            record=None,
            profile=profile,
            requested_model=requested_model,
            provider=registered,
            provider_model_id=requested_model,
        )

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
