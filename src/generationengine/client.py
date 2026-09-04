"""Public GenerationEngine execution facade."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from generationengine.catalog import Capability
from generationengine.failures import FailureCode, InferenceFailure
from generationengine.observation import InferenceObservation, ObservationState
from generationengine.providers.base import (
    ImageProvider,
    TextCompleted,
    TextFailed,
    TextGenerationCall,
    TextProvider,
    TextStreamEvent,
)
from generationengine.providers.errors import ProviderError
from generationengine.resolver import ResolutionError, resolve
from generationengine.types import (
    GeneratedImage,
    GenerationEngineError,
    ImageRequest,
    ImageResult,
    TextRequest,
    TextResult,
)

MAX_ATTEMPTS = 3
DEFAULT_ATTEMPT_TIMEOUT_S = 60.0


class GenerationClient:
    """Product-neutral inference client. Provider SDKs stay behind adapters."""

    def __init__(
        self,
        *,
        text_provider: TextProvider | None = None,
        image_provider: ImageProvider | None = None,
    ) -> None:
        self._text = text_provider
        self._image = image_provider

    @classmethod
    def from_env(cls) -> GenerationClient:
        """Construct providers from extras/env. Missing extras fail at first use."""
        return cls()

    def _text_provider(self) -> TextProvider:
        if self._text is None:
            from generationengine.providers.openai_text import OpenAITextProvider

            try:
                self._text = OpenAITextProvider()
            except ProviderError as exc:
                raise _config_error(exc.failure, provider="openai") from exc
        return self._text

    def _image_provider(self) -> ImageProvider:
        if self._image is None:
            from generationengine.providers.fal_provider import FalProvider

            try:
                self._image = FalProvider()
            except (ImportError, ValueError) as exc:
                raise _config_error(
                    InferenceFailure.from_code(
                        FailureCode.CONFIGURATION_UNAVAILABLE,
                        str(exc) or "fal extra or FAL_KEY is required for image generation.",
                    ),
                    provider="fal",
                ) from exc
        return self._image

    async def generate_text(self, request: TextRequest) -> TextResult:
        return await self._generate_text(request, capability=Capability.TEXT)

    async def generate_structured(self, request: TextRequest) -> TextResult:
        if not request.json_schema:
            raise _config_error(
                InferenceFailure.from_code(
                    FailureCode.INVALID_REQUEST,
                    "generate_structured requires json_schema.",
                )
            )
        return await self._generate_text(request, capability=Capability.STRUCTURED_TEXT)

    async def stream_text(self, request: TextRequest) -> AsyncIterator[TextStreamEvent]:
        started = time.monotonic()
        try:
            resolution = resolve(
                capability=Capability.STREAMING_TEXT,
                profile=request.profile,
                model=request.model,
            )
        except ResolutionError as exc:
            yield TextFailed(
                failure=exc.failure,
                observation=_failed_observation(
                    failure=exc.failure,
                    request=request,
                    latency_ms=_elapsed_ms(started),
                    retry_count=0,
                ),
            )
            return
        provider = self._text_provider()
        call = _text_call(request, resolution.record.provider_model_id)
        terminal = False
        async for event in provider.stream(call):
            if isinstance(event, (TextCompleted, TextFailed)):
                terminal = True
            yield event
        if not terminal:
            failure = InferenceFailure.from_code(
                FailureCode.STREAM_INCOMPLETE,
                "Text stream ended without TextCompleted or TextFailed.",
            )
            yield TextFailed(
                failure=failure,
                observation=_failed_observation(
                    failure=failure,
                    request=request,
                    resolution=resolution,
                    latency_ms=_elapsed_ms(started),
                    retry_count=0,
                ),
            )

    async def generate_image(self, request: ImageRequest) -> ImageResult:
        return await self._generate_image(request, capability=Capability.IMAGE)

    async def edit_image(self, request: ImageRequest) -> ImageResult:
        if not request.base_image_base64 and not request.source_image_url:
            raise _config_error(
                InferenceFailure.from_code(
                    FailureCode.INVALID_REQUEST,
                    "edit_image requires base_image_base64 or source_image_url.",
                )
            )
        return await self._generate_image(request, capability=Capability.IMAGE_EDIT)

    async def _generate_text(
        self,
        request: TextRequest,
        *,
        capability: Capability,
    ) -> TextResult:
        started = time.monotonic()
        try:
            resolution = resolve(
                capability=capability,
                profile=request.profile,
                model=request.model,
            )
        except ResolutionError as exc:
            raise _config_error(exc.failure, request=request) from exc
        provider = self._text_provider()
        call = _text_call(request, resolution.record.provider_model_id)
        retry_count = 0
        last_error: ProviderError | None = None
        timeout_s = _timeout_s(request.deadline_ms)
        for attempt in range(MAX_ATTEMPTS):
            try:
                result = await _with_timeout(provider.generate(call), timeout_s)
                observation = _completed_observation(
                    request=request,
                    resolution=resolution,
                    result=result,
                    latency_ms=_elapsed_ms(started),
                    retry_count=retry_count,
                )
                if result.refused:
                    failure = InferenceFailure.from_code(
                        FailureCode.PROVIDER_REFUSED,
                        "Provider refused the request.",
                    )
                    raise GenerationEngineError(
                        failure,
                        observation.model_copy(
                            update={
                                "state": ObservationState.REFUSED,
                                "failure_code": FailureCode.PROVIDER_REFUSED,
                            }
                        ),
                    )
                return TextResult(text=result.text, parsed=result.parsed, observation=observation)
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == MAX_ATTEMPTS - 1:
                    retry_count = attempt
                    raise GenerationEngineError(
                        exc.failure,
                        _failed_observation(
                            failure=exc.failure,
                            request=request,
                            resolution=resolution,
                            latency_ms=_elapsed_ms(started),
                            retry_count=retry_count,
                            result=exc,
                        ),
                    ) from exc
                retry_count = attempt + 1
            except TimeoutError as exc:
                last_error = ProviderError.from_code(
                    FailureCode.PROVIDER_TIMEOUT,
                    f"Request timed out after {timeout_s}s",
                )
                if attempt == MAX_ATTEMPTS - 1:
                    raise GenerationEngineError(
                        last_error.failure,
                        _failed_observation(
                            failure=last_error.failure,
                            request=request,
                            resolution=resolution,
                            latency_ms=_elapsed_ms(started),
                            retry_count=attempt,
                        ),
                    ) from exc
                retry_count = attempt + 1
        assert last_error is not None
        raise GenerationEngineError(
            last_error.failure,
            _failed_observation(
                failure=last_error.failure,
                request=request,
                resolution=resolution,
                latency_ms=_elapsed_ms(started),
                retry_count=retry_count,
            ),
        )

    async def _generate_image(
        self,
        request: ImageRequest,
        *,
        capability: Capability,
    ) -> ImageResult:
        started = time.monotonic()
        try:
            resolution = resolve(
                capability=capability,
                profile=request.profile,
                model=request.model,
            )
        except ResolutionError as exc:
            raise _config_error(exc.failure, image=request) from exc
        provider = self._image_provider()
        retry_count = 0
        timeout_s = _timeout_s(request.deadline_ms)
        last_error: ProviderError | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                blobs = await _with_timeout(
                    provider.generate(
                        prompt=request.prompt,
                        model=resolution.record.provider_model_id,
                        num_images=request.num_images,
                        size=(request.width, request.height),
                        image_url=request.source_image_url,
                        strength=request.strength if request.strength is not None else 0.85,
                        mask_base64=request.mask_base64,
                        base_image_base64=request.base_image_base64,
                        negative_prompt=request.negative_prompt,
                    ),
                    timeout_s,
                )
                observation = InferenceObservation(
                    provider=resolution.record.provider,
                    requested_profile=request.profile.value if request.profile else None,
                    requested_model=request.model,
                    resolved_model=resolution.catalog_id,
                    latency_ms=_elapsed_ms(started),
                    retry_count=retry_count,
                    state=ObservationState.COMPLETED,
                )
                images = [
                    GeneratedImage(
                        content=blob,
                        media_type="image/png",
                        width=request.width,
                        height=request.height,
                    )
                    for blob in blobs
                ]
                return ImageResult(images=images, observation=observation)
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == MAX_ATTEMPTS - 1:
                    raise GenerationEngineError(
                        exc.failure,
                        _failed_observation(
                            failure=exc.failure,
                            image=request,
                            resolution=resolution,
                            latency_ms=_elapsed_ms(started),
                            retry_count=attempt,
                        ),
                    ) from exc
                retry_count = attempt + 1
            except Exception as exc:
                mapped = _map_image_exception(exc)
                last_error = mapped
                if not mapped.retryable or attempt == MAX_ATTEMPTS - 1:
                    raise GenerationEngineError(
                        mapped.failure,
                        _failed_observation(
                            failure=mapped.failure,
                            image=request,
                            resolution=resolution,
                            latency_ms=_elapsed_ms(started),
                            retry_count=attempt,
                        ),
                    ) from exc
                retry_count = attempt + 1
        assert last_error is not None
        raise GenerationEngineError(
            last_error.failure,
            _failed_observation(
                failure=last_error.failure,
                image=request,
                resolution=resolution,
                latency_ms=_elapsed_ms(started),
                retry_count=retry_count,
            ),
        )


def _text_call(request: TextRequest, model: str) -> TextGenerationCall:
    return TextGenerationCall(
        model=model,
        user_prompt=request.user_prompt,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        json_schema=request.json_schema,
        schema_name=request.schema_name,
    )


async def _with_timeout(awaitable, timeout_s: float):
    import asyncio

    return await asyncio.wait_for(awaitable, timeout=timeout_s)


def _timeout_s(deadline_ms: int | None) -> float:
    if deadline_ms is None:
        return DEFAULT_ATTEMPT_TIMEOUT_S
    return max(deadline_ms / 1000.0, 0.001)


def _elapsed_ms(started: float) -> int:
    return max(int((time.monotonic() - started) * 1000), 0)


def _config_error(
    failure: InferenceFailure,
    *,
    provider: str | None = None,
    request: TextRequest | None = None,
    image: ImageRequest | None = None,
) -> GenerationEngineError:
    return GenerationEngineError(
        failure,
        _failed_observation(
            failure=failure,
            request=request,
            image=image,
            provider=provider,
            latency_ms=0,
            retry_count=0,
        ),
    )


def _completed_observation(
    *,
    request: TextRequest,
    resolution,
    result,
    latency_ms: int,
    retry_count: int,
) -> InferenceObservation:
    return InferenceObservation(
        provider=resolution.record.provider,
        requested_profile=request.profile.value if request.profile else None,
        requested_model=request.model,
        resolved_model=resolution.catalog_id,
        response_model=result.response_model,
        provider_request_id=result.provider_request_id,
        input_tokens=result.input_tokens,
        cached_input_tokens=result.cached_input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=None,
        latency_ms=latency_ms,
        retry_count=retry_count,
        state=ObservationState.COMPLETED,
        pricing_source=resolution.record.pricing_source,
    )


def _failed_observation(
    *,
    failure: InferenceFailure,
    request: TextRequest | None = None,
    image: ImageRequest | None = None,
    resolution=None,
    latency_ms: int,
    retry_count: int,
    result: ProviderError | None = None,
    provider: str | None = None,
) -> InferenceObservation:
    profile = None
    model = None
    if request is not None:
        profile = request.profile.value if request.profile else None
        model = request.model
    elif image is not None:
        profile = image.profile.value if image.profile else None
        model = image.model
    state = ObservationState.FAILED
    if failure.code is FailureCode.PROVIDER_REFUSED:
        state = ObservationState.REFUSED
    elif failure.code in {
        FailureCode.STREAM_INCOMPLETE,
        FailureCode.STRUCTURED_OUTPUT_INVALID,
        FailureCode.MALFORMED_PROVIDER_RESPONSE,
    }:
        state = ObservationState.INCOMPLETE
    return InferenceObservation(
        provider=provider or (resolution.record.provider if resolution else None),
        requested_profile=profile,
        requested_model=model,
        resolved_model=resolution.catalog_id if resolution else None,
        response_model=result.response_model if result else None,
        provider_request_id=result.provider_request_id if result else None,
        input_tokens=result.input_tokens if result else None,
        cached_input_tokens=result.cached_input_tokens if result else None,
        output_tokens=result.output_tokens if result else None,
        latency_ms=latency_ms,
        retry_count=retry_count,
        state=state,
        failure_code=failure.code,
        pricing_source=resolution.record.pricing_source if resolution else None,
    )


def _map_image_exception(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    return ProviderError.from_code(FailureCode.PROVIDER_ERROR, str(exc) or type(exc).__name__)
