"""Public GenerationEngine execution facade."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

from generationengine.catalog import Capability
from generationengine.conformance import (
    ConformanceError,
    candidate_from_result,
    check_schema,
    repair_instruction,
    validate_against_schema,
)
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
MAX_CONFORMANCE_RETRIES = 1
DEFAULT_DEADLINE_S = 60.0
BACKOFF_SECONDS = (0.5, 1.0)

T = TypeVar("T")


class GenerationClient:
    """Product-neutral inference client. Provider SDKs stay behind adapters."""

    def __init__(
        self,
        *,
        text_provider: TextProvider | None = None,
        image_provider: ImageProvider | None = None,
        text_providers: dict[str, TextProvider] | None = None,
    ) -> None:
        self._text_providers: dict[str, TextProvider] = {}
        if text_providers:
            for name, provider in text_providers.items():
                self._text_providers[name.strip().lower()] = provider
        if text_provider is not None:
            self._text_providers.setdefault("openai", text_provider)
        self._image = image_provider

    @classmethod
    def from_env(cls) -> GenerationClient:
        """Construct providers from extras/env. Missing extras fail at first use."""
        return cls()

    def _text_provider_for(self, provider_id: str) -> TextProvider:
        key = provider_id.strip().lower()
        cached = self._text_providers.get(key)
        if cached is not None:
            return cached
        if key == "openai":
            from generationengine.providers.openai_text import OpenAITextProvider

            try:
                provider = OpenAITextProvider()
            except ProviderError as exc:
                raise _config_error(exc.failure, provider="openai") from exc
            self._text_providers[key] = provider
            return provider
        if key == "openrouter":
            from generationengine.providers.openrouter_text import OpenRouterTextProvider

            try:
                provider = OpenRouterTextProvider()
            except ProviderError as exc:
                raise _config_error(exc.failure, provider="openrouter") from exc
            self._text_providers[key] = provider
            return provider
        raise _config_error(
            InferenceFailure.from_code(
                FailureCode.UNSUPPORTED_CAPABILITY,
                f"Text provider {key!r} is not registered.",
            ),
            provider=key,
        )

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
        if request.json_object and request.json_schema is not None:
            raise _config_error(
                InferenceFailure.from_code(
                    FailureCode.INVALID_REQUEST,
                    "json_object and json_schema are mutually exclusive.",
                ),
                request=request,
                provider_attempt_count=0,
            )
        return await self._generate_text(request, capability=Capability.TEXT)

    async def generate_structured(self, request: TextRequest) -> TextResult:
        if request.json_object:
            raise _config_error(
                InferenceFailure.from_code(
                    FailureCode.INVALID_REQUEST,
                    "generate_structured does not accept json_object mode.",
                ),
                request=request,
                provider_attempt_count=0,
            )
        if not request.json_schema:
            raise _config_error(
                InferenceFailure.from_code(
                    FailureCode.INVALID_REQUEST,
                    "generate_structured requires json_schema.",
                ),
                request=request,
                provider_attempt_count=0,
            )
        try:
            check_schema(request.json_schema)
        except ConformanceError as exc:
            raise _config_error(
                InferenceFailure.from_code(
                    FailureCode.INVALID_REQUEST,
                    "generate_structured requires a valid JSON Schema object.",
                ),
                request=request,
                provider_attempt_count=0,
            ) from exc
        return await self._generate_structured(request)

    async def stream_text(self, request: TextRequest) -> AsyncIterator[TextStreamEvent]:
        """Yield deltas, then exactly one terminal. Streaming does not retry."""
        started = time.monotonic()
        if request.json_object:
            yield _stream_failure(
                failure=InferenceFailure.from_code(
                    FailureCode.INVALID_REQUEST,
                    "stream_text does not accept json_object mode.",
                ),
                request=request,
                started=started,
            )
            return
        deadline_s = _deadline_s(request.deadline_ms)
        resolution = None
        try:
            resolution = resolve(
                capability=Capability.STREAMING_TEXT,
                profile=request.profile,
                model=request.model,
                provider=request.provider,
            )
        except ResolutionError as exc:
            yield _stream_failure(
                failure=exc.failure,
                request=request,
                started=started,
            )
            return

        stream = None
        terminal = False
        try:
            try:
                provider = self._text_provider_for(resolution.provider)
            except GenerationEngineError as exc:
                yield _stream_failure(
                    failure=exc.failure,
                    request=request,
                    resolution=resolution,
                    started=started,
                    provider=exc.observation.provider,
                )
                return
            call = _text_call(request, resolution.provider_model_id)
            stream = provider.stream(call)
            iterator = stream.__aiter__()
            while True:
                remaining = _remaining_s(started, deadline_s)
                if remaining <= 0:
                    raise TimeoutError()
                event = await _with_timeout(_anext_or_none(iterator), remaining)
                if event is None:
                    break
                if isinstance(event, (TextCompleted, TextFailed)):
                    if terminal:
                        continue
                    terminal = True
                    yield _public_stream_terminal(
                        event,
                        request=request,
                        resolution=resolution,
                        latency_ms=_elapsed_ms(started),
                    )
                    return
                yield event
            if not terminal:
                yield _stream_failure(
                    failure=InferenceFailure.from_code(
                        FailureCode.STREAM_INCOMPLETE,
                        "Text stream ended without TextCompleted or TextFailed.",
                    ),
                    request=request,
                    resolution=resolution,
                    started=started,
                )
        except TimeoutError:
            if not terminal:
                yield _stream_failure(
                    failure=InferenceFailure.from_code(FailureCode.PROVIDER_TIMEOUT),
                    request=request,
                    resolution=resolution,
                    started=started,
                )
        except ProviderError as exc:
            if not terminal:
                yield _stream_failure(
                    failure=exc.failure,
                    request=request,
                    resolution=resolution,
                    started=started,
                    result=exc,
                )
        except GenerationEngineError as exc:
            if not terminal:
                yield _stream_failure(
                    failure=exc.failure,
                    request=request,
                    resolution=resolution,
                    started=started,
                    provider=exc.observation.provider,
                )
        except Exception as exc:
            if not terminal:
                mapped = _map_provider_exception(exc)
                yield _stream_failure(
                    failure=mapped.failure,
                    request=request,
                    resolution=resolution,
                    started=started,
                    result=mapped,
                )
        finally:
            await _aclose_stream(stream)

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
        deadline_s = _deadline_s(request.deadline_ms)
        try:
            resolution = resolve(
                capability=capability,
                profile=request.profile,
                model=request.model,
                provider=request.provider,
            )
        except ResolutionError as exc:
            raise _config_error(exc.failure, request=request) from exc
        provider = self._text_provider_for(resolution.provider)
        call = _text_call(request, resolution.provider_model_id)
        try:
            result, retry_count = await _execute_with_retries(
                started=started,
                deadline_s=deadline_s,
                attempt=lambda: provider.generate(call),
            )
        except ProviderError as exc:
            raise GenerationEngineError(
                exc.failure,
                _failed_observation(
                    failure=exc.failure,
                    request=request,
                    resolution=resolution,
                    latency_ms=_elapsed_ms(started),
                    retry_count=_retry_count_from_error(exc),
                    result=exc,
                ),
            ) from exc
        observation = _completed_observation(
            request=request,
            resolution=resolution,
            result=result,
            latency_ms=_elapsed_ms(started),
            retry_count=retry_count,
            conformance_retry_count=0,
            provider_attempt_count=1 + retry_count,
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
        return TextResult(
            text=result.text,
            parsed=None if request.json_object else result.parsed,
            observation=observation,
        )

    async def _generate_structured(self, request: TextRequest) -> TextResult:
        started = time.monotonic()
        deadline_s = _deadline_s(request.deadline_ms)
        try:
            resolution = resolve(
                capability=Capability.STRUCTURED_TEXT,
                profile=request.profile,
                model=request.model,
                provider=request.provider,
            )
        except ResolutionError as exc:
            raise _config_error(exc.failure, request=request) from exc
        provider = self._text_provider_for(resolution.provider)
        schema = request.json_schema
        assert schema is not None

        usage_results: list = []
        transport_retries = 0
        conformance_retries = 0
        provider_attempts = 0
        call = _text_call(request, resolution.provider_model_id)

        while True:
            if _remaining_s(started, deadline_s) <= 0:
                raise GenerationEngineError(
                    InferenceFailure.from_code(FailureCode.PROVIDER_TIMEOUT),
                    _structured_observation(
                        request=request,
                        resolution=resolution,
                        latency_ms=_elapsed_ms(started),
                        transport_retries=transport_retries,
                        conformance_retries=conformance_retries,
                        provider_attempts=provider_attempts,
                        usage_results=usage_results,
                        failure=InferenceFailure.from_code(FailureCode.PROVIDER_TIMEOUT),
                    ),
                )
            try:
                result, call_retries = await _execute_with_retries(
                    started=started,
                    deadline_s=deadline_s,
                    attempt=lambda current=call: provider.generate(current),
                    attempt_usage=usage_results,
                )
            except ProviderError as exc:
                failed_retries = _retry_count_from_error(exc)
                raise GenerationEngineError(
                    exc.failure,
                    _structured_observation(
                        request=request,
                        resolution=resolution,
                        latency_ms=_elapsed_ms(started),
                        transport_retries=transport_retries + failed_retries,
                        conformance_retries=conformance_retries,
                        provider_attempts=provider_attempts + 1 + failed_retries,
                        usage_results=usage_results,
                        failure=exc.failure,
                        result=exc,
                    ),
                ) from exc
            transport_retries += call_retries
            provider_attempts += 1 + call_retries
            if result.refused:
                failure = InferenceFailure.from_code(
                    FailureCode.PROVIDER_REFUSED,
                    "Provider refused the request.",
                )
                raise GenerationEngineError(
                    failure,
                    _structured_observation(
                        request=request,
                        resolution=resolution,
                        latency_ms=_elapsed_ms(started),
                        transport_retries=transport_retries,
                        conformance_retries=conformance_retries,
                        provider_attempts=provider_attempts,
                        usage_results=usage_results,
                        failure=failure,
                        result=result,
                    ),
                )
            try:
                parsed = candidate_from_result(result)
                validate_against_schema(parsed, schema)
            except ConformanceError as failure:
                if conformance_retries >= MAX_CONFORMANCE_RETRIES:
                    invalid = InferenceFailure.from_code(
                        FailureCode.STRUCTURED_OUTPUT_INVALID,
                        "Structured output did not satisfy the supplied schema.",
                    )
                    raise GenerationEngineError(
                        invalid,
                        _structured_observation(
                            request=request,
                            resolution=resolution,
                            latency_ms=_elapsed_ms(started),
                            transport_retries=transport_retries,
                            conformance_retries=conformance_retries,
                            provider_attempts=provider_attempts,
                            usage_results=usage_results,
                            failure=invalid,
                            result=result,
                        ),
                    ) from None
                if _remaining_s(started, deadline_s) <= 0:
                    timeout = InferenceFailure.from_code(FailureCode.PROVIDER_TIMEOUT)
                    raise GenerationEngineError(
                        timeout,
                        _structured_observation(
                            request=request,
                            resolution=resolution,
                            latency_ms=_elapsed_ms(started),
                            transport_retries=transport_retries,
                            conformance_retries=conformance_retries,
                            provider_attempts=provider_attempts,
                            usage_results=usage_results,
                            failure=timeout,
                            result=result,
                        ),
                    ) from None
                conformance_retries += 1
                call = _text_call(
                    request,
                    resolution.provider_model_id,
                    user_prompt=f"{request.user_prompt}\n\n{repair_instruction(failure)}",
                )
                continue
            usage = _aggregate_usage(usage_results)
            observation = InferenceObservation(
                provider=resolution.provider,
                requested_profile=request.profile.value if request.profile else None,
                requested_model=request.model,
                resolved_model=resolution.resolved_model,
                response_model=result.response_model,
                provider_request_id=result.provider_request_id,
                provider_response_id=getattr(result, "provider_response_id", None),
                input_tokens=usage["input_tokens"],
                cached_input_tokens=usage["cached_input_tokens"],
                output_tokens=usage["output_tokens"],
                cost_usd=None,
                latency_ms=_elapsed_ms(started),
                retry_count=transport_retries,
                transport_retry_count=transport_retries,
                conformance_retry_count=conformance_retries,
                provider_attempt_count=provider_attempts,
                state=ObservationState.COMPLETED,
                pricing_source=resolution.pricing_source,
            )
            return TextResult(text=result.text, parsed=parsed, observation=observation)

    async def _generate_image(
        self,
        request: ImageRequest,
        *,
        capability: Capability,
    ) -> ImageResult:
        started = time.monotonic()
        deadline_s = _deadline_s(request.deadline_ms)
        try:
            resolution = resolve(
                capability=capability,
                profile=request.profile,
                model=request.model,
            )
        except ResolutionError as exc:
            raise _config_error(exc.failure, image=request) from exc
        provider = self._image_provider()

        async def _once() -> list[bytes]:
            try:
                return await provider.generate(
                    prompt=request.prompt,
                    model=resolution.provider_model_id,
                    num_images=request.num_images,
                    size=(request.width, request.height),
                    image_url=request.source_image_url,
                    strength=request.strength if request.strength is not None else 0.85,
                    mask_base64=request.mask_base64,
                    base_image_base64=request.base_image_base64,
                    negative_prompt=request.negative_prompt,
                )
            except ProviderError:
                raise
            except TimeoutError:
                raise
            except Exception as exc:
                raise _map_provider_exception(exc) from exc

        try:
            blobs, retry_count = await _execute_with_retries(
                started=started,
                deadline_s=deadline_s,
                attempt=_once,
            )
        except ProviderError as exc:
            raise GenerationEngineError(
                exc.failure,
                _failed_observation(
                    failure=exc.failure,
                    image=request,
                    resolution=resolution,
                    latency_ms=_elapsed_ms(started),
                    retry_count=_retry_count_from_error(exc),
                    result=exc,
                ),
            ) from exc
        observation = InferenceObservation(
            provider=resolution.provider,
            requested_profile=request.profile.value if request.profile else None,
            requested_model=request.model,
            resolved_model=resolution.resolved_model,
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


def _text_call(
    request: TextRequest,
    model: str,
    user_prompt: str | None = None,
) -> TextGenerationCall:
    return TextGenerationCall(
        model=model,
        user_prompt=request.user_prompt if user_prompt is None else user_prompt,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        json_object=request.json_object,
        json_schema=request.json_schema,
        schema_name=request.schema_name,
    )


async def _execute_with_retries(
    *,
    started: float,
    deadline_s: float,
    attempt: Callable[[], Awaitable[T]],
    attempt_usage: list | None = None,
) -> tuple[T, int]:
    """Run one provider operation under a single overall deadline.

    `deadline_s` bounds the whole GenerationEngine call, including backoff.
    Each attempt is limited to remaining budget. Provider SDK retries are not
    this loop; adapters must disable them.

    When `attempt_usage` is provided, every provider generate outcome is
    appended — including retryable errors that later recover — so callers can
    aggregate usage across the whole operation.
    """

    def _record(item: object) -> None:
        if attempt_usage is not None:
            attempt_usage.append(item)

    last_error: ProviderError | None = None
    retry_count = 0
    for attempt_index in range(MAX_ATTEMPTS):
        remaining = _remaining_s(started, deadline_s)
        if remaining <= 0:
            last_error = _timeout_error(deadline_s, retry_count=retry_count)
            break
        try:
            result = await _with_timeout(attempt(), remaining)
            _record(result)
            return result, retry_count
        except ProviderError as exc:
            _record(exc)
            last_error = exc
            last_error.retry_count = retry_count
        except TimeoutError:
            last_error = _timeout_error(deadline_s, retry_count=retry_count)
            _record(last_error)
        if not last_error.retryable or attempt_index == MAX_ATTEMPTS - 1:
            raise last_error
        delay = BACKOFF_SECONDS[min(attempt_index, len(BACKOFF_SECONDS) - 1)]
        remaining_after = _remaining_s(started, deadline_s)
        if remaining_after <= delay:
            raise last_error
        await asyncio.sleep(delay)
        retry_count = attempt_index + 1
    assert last_error is not None
    raise last_error


async def _with_timeout(awaitable: Awaitable[T], timeout_s: float) -> T:
    return await asyncio.wait_for(awaitable, timeout=timeout_s)


async def _anext_or_none(iterator: AsyncIterator[T]) -> T | None:
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return None


async def _aclose_stream(stream: AsyncIterator[T] | None) -> None:
    if stream is None:
        return
    aclose = getattr(stream, "aclose", None)
    if not callable(aclose):
        return
    try:
        await aclose()
    except Exception:
        return


def _deadline_s(deadline_ms: int | None) -> float:
    if deadline_ms is None:
        return DEFAULT_DEADLINE_S
    return max(deadline_ms / 1000.0, 0.001)


def _remaining_s(started: float, deadline_s: float) -> float:
    return deadline_s - (time.monotonic() - started)


def _timeout_error(_deadline_s: float, *, retry_count: int) -> ProviderError:
    error = ProviderError.from_code(FailureCode.PROVIDER_TIMEOUT)
    error.retry_count = retry_count
    return error


def _retry_count_from_error(exc: ProviderError) -> int:
    return getattr(exc, "retry_count", 0)


def _elapsed_ms(started: float) -> int:
    return max(int((time.monotonic() - started) * 1000), 0)


def _aggregate_usage(results: list) -> dict[str, int | None]:
    """Sum known attempt usage. Any unknown field makes that aggregate None."""

    def _field(name: str) -> int | None:
        values = [getattr(item, name, None) for item in results]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    return {
        "input_tokens": _field("input_tokens"),
        "cached_input_tokens": _field("cached_input_tokens"),
        "output_tokens": _field("output_tokens"),
    }


def _structured_observation(
    *,
    request: TextRequest,
    resolution,
    latency_ms: int,
    transport_retries: int,
    conformance_retries: int,
    provider_attempts: int,
    usage_results: list,
    failure: InferenceFailure,
    result=None,
) -> InferenceObservation:
    items = list(usage_results)
    if isinstance(result, ProviderError) and result not in items:
        items.append(result)
    usage = _aggregate_usage(items)
    terminal = result
    return _failed_observation(
        failure=failure,
        request=request,
        resolution=resolution,
        latency_ms=latency_ms,
        retry_count=transport_retries,
        result=terminal if isinstance(terminal, ProviderError) else None,
        provider_request_id=getattr(terminal, "provider_request_id", None),
        provider_response_id=getattr(terminal, "provider_response_id", None),
        response_model=getattr(terminal, "response_model", None),
        usage=usage,
        conformance_retry_count=conformance_retries,
        provider_attempt_count=provider_attempts,
    )


def _config_error(
    failure: InferenceFailure,
    *,
    provider: str | None = None,
    request: TextRequest | None = None,
    image: ImageRequest | None = None,
    provider_attempt_count: int | None = None,
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
            provider_attempt_count=provider_attempt_count,
        ),
    )


def _stream_failure(
    *,
    failure: InferenceFailure,
    request: TextRequest,
    started: float,
    resolution=None,
    provider: str | None = None,
    result: ProviderError | None = None,
) -> TextFailed:
    return TextFailed(
        failure=failure,
        observation=_failed_observation(
            failure=failure,
            request=request,
            resolution=resolution,
            latency_ms=_elapsed_ms(started),
            retry_count=0,
            result=result,
            provider=provider,
        ),
    )


def _public_stream_terminal(
    event: TextCompleted | TextFailed,
    *,
    request: TextRequest,
    resolution,
    latency_ms: int,
) -> TextCompleted | TextFailed:
    provider_obs = event.observation
    if isinstance(event, TextCompleted):
        return TextCompleted(
            final_text=event.final_text,
            observation=_completed_observation_from_stream(
                request=request,
                resolution=resolution,
                provider_obs=provider_obs,
                latency_ms=latency_ms,
            ),
        )
    return TextFailed(
        failure=_public_failure(event.failure),
        observation=_failed_observation(
            failure=event.failure,
            request=request,
            resolution=resolution,
            latency_ms=latency_ms,
            retry_count=0,
            provider_request_id=provider_obs.provider_request_id,
            provider_response_id=provider_obs.provider_response_id,
            response_model=provider_obs.response_model,
            input_tokens=provider_obs.input_tokens,
            cached_input_tokens=provider_obs.cached_input_tokens,
            output_tokens=provider_obs.output_tokens,
        ),
    )


def _completed_observation_from_stream(
    *,
    request: TextRequest,
    resolution,
    provider_obs: InferenceObservation,
    latency_ms: int,
) -> InferenceObservation:
    return InferenceObservation(
        provider=resolution.provider,
        requested_profile=request.profile.value if request.profile else None,
        requested_model=request.model,
        resolved_model=resolution.resolved_model,
        response_model=provider_obs.response_model,
        provider_request_id=provider_obs.provider_request_id,
        provider_response_id=provider_obs.provider_response_id,
        input_tokens=provider_obs.input_tokens,
        cached_input_tokens=provider_obs.cached_input_tokens,
        output_tokens=provider_obs.output_tokens,
        cost_usd=None,
        latency_ms=latency_ms,
        retry_count=0,
        state=ObservationState.COMPLETED,
        pricing_source=resolution.pricing_source,
        conformance_retry_count=0,
        provider_attempt_count=1,
        transport_retry_count=0,
    )


def _completed_observation(
    *,
    request: TextRequest,
    resolution,
    result,
    latency_ms: int,
    retry_count: int,
    conformance_retry_count: int = 0,
    provider_attempt_count: int | None = None,
) -> InferenceObservation:
    return InferenceObservation(
        provider=resolution.provider,
        requested_profile=request.profile.value if request.profile else None,
        requested_model=request.model,
        resolved_model=resolution.resolved_model,
        response_model=result.response_model,
        provider_request_id=result.provider_request_id,
        provider_response_id=getattr(result, "provider_response_id", None),
        input_tokens=result.input_tokens,
        cached_input_tokens=result.cached_input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=None,
        latency_ms=latency_ms,
        retry_count=retry_count,
        transport_retry_count=retry_count,
        conformance_retry_count=conformance_retry_count,
        provider_attempt_count=provider_attempt_count if provider_attempt_count is not None else 1 + retry_count,
        state=ObservationState.COMPLETED,
        pricing_source=resolution.pricing_source,
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
    provider_request_id: str | None = None,
    provider_response_id: str | None = None,
    response_model: str | None = None,
    input_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    output_tokens: int | None = None,
    conformance_retry_count: int = 0,
    provider_attempt_count: int | None = None,
    usage: dict[str, int | None] | None = None,
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
        provider=provider or (resolution.provider if resolution else None),
        requested_profile=profile,
        requested_model=model,
        resolved_model=resolution.resolved_model if resolution else None,
        response_model=response_model or (result.response_model if result else None),
        provider_request_id=provider_request_id
        or (result.provider_request_id if result else None),
        provider_response_id=provider_response_id
        or (getattr(result, "provider_response_id", None) if result else None),
        input_tokens=(
            usage["input_tokens"]
            if usage is not None
            else (input_tokens if input_tokens is not None else (result.input_tokens if result else None))
        ),
        cached_input_tokens=(
            usage["cached_input_tokens"]
            if usage is not None
            else (
                cached_input_tokens
                if cached_input_tokens is not None
                else (result.cached_input_tokens if result else None)
            )
        ),
        output_tokens=(
            usage["output_tokens"]
            if usage is not None
            else (output_tokens if output_tokens is not None else (result.output_tokens if result else None))
        ),
        latency_ms=latency_ms,
        retry_count=retry_count,
        transport_retry_count=retry_count,
        conformance_retry_count=conformance_retry_count,
        provider_attempt_count=provider_attempt_count,
        state=state,
        failure_code=failure.code,
        pricing_source=resolution.pricing_source if resolution else None,
    )


def _public_failure(failure: InferenceFailure) -> InferenceFailure:
    return InferenceFailure.from_code(failure.code, failure.message)


def _map_provider_exception(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        rebuilt = ProviderError.from_code(
            exc.failure.code,
            exc.failure.message,
            provider_request_id=exc.provider_request_id,
            provider_response_id=exc.provider_response_id,
            response_model=exc.response_model,
            input_tokens=exc.input_tokens,
            cached_input_tokens=exc.cached_input_tokens,
            output_tokens=exc.output_tokens,
        )
        rebuilt.retry_count = exc.retry_count
        return rebuilt
    if isinstance(exc, TimeoutError):
        return ProviderError.from_code(FailureCode.PROVIDER_TIMEOUT)
    return ProviderError.from_code(FailureCode.PROVIDER_ERROR)
