"""OpenAI text/structured/stream adapter. SDK types stay inside this module."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

from generationengine.failures import FailureCode, InferenceFailure, Retryability
from generationengine.observation import InferenceObservation, ObservationState
from generationengine.providers.base import (
    TextCompleted,
    TextDelta,
    TextFailed,
    TextGenerationCall,
    TextGenerationResult,
    TextStreamEvent,
)
from generationengine.providers.errors import ProviderError
from generationengine.providers.openai_compatible import (
    AsyncOpenAI,
    map_openai_compatible_exception,
    require_openai_sdk,
)
from generationengine.text_dispatch import TextDispatchGate
from generationengine.utils.schema_utils import make_schema_strict


class OpenAITextProvider:
    def __init__(
        self,
        client: Any | None = None,
        api_key: str | None = None,
        *,
        dispatch_gate: TextDispatchGate | None = None,
        transport: Any | None = None,
    ) -> None:
        self._guarded = dispatch_gate is not None
        if (client is not None and dispatch_gate is not None) or (
            transport is not None and dispatch_gate is None
        ):
            raise ValueError("Guarded dispatch requires the official SDK and an explicit gate")
        if client is not None:
            self._client = client
            return
        require_openai_sdk(extra_name="openai", client_cls=AsyncOpenAI)
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderError.from_code(
                FailureCode.CONFIGURATION_UNAVAILABLE,
                "OPENAI_API_KEY is required for text generation.",
            )
        if dispatch_gate is not None:
            import httpx

            from generationengine.providers.openai_dispatch import GuardedOpenAITransport

            wire = GuardedOpenAITransport(
                dispatch_gate, transport or httpx.AsyncHTTPTransport(retries=0)
            )
            http = httpx.AsyncClient(transport=wire, follow_redirects=False, trust_env=False)
            self._client = AsyncOpenAI(
                api_key=key,
                max_retries=0,
                base_url="https://api.openai.com/v1",
                http_client=http,
            )
        else:
            self._client = AsyncOpenAI(api_key=key, max_retries=0)

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        if self._guarded and call.json_schema is not None and call.max_conformance_retries != 0:
            raise ProviderError.from_code(
                FailureCode.INVALID_REQUEST,
                "Guarded text dispatch does not support structural repair.",
            )
        kwargs = self._request_kwargs(call)
        if self._guarded:
            kwargs.update(tools=[], stream=False)
        try:
            response = await self._client.responses.create(**kwargs)
        except Exception as exc:
            mapped = self._map_exception(exc)
            if self._guarded:
                mapped.failure = mapped.failure.model_copy(update={"retryability": Retryability.NO})
            raise mapped from exc
        if self._guarded:
            return self._guarded_result(response)
        if getattr(response, "status", None) == "incomplete" and not getattr(
            response, "refusal", None
        ):
            request_id, response_id = _ids_from_response(response)
            usage = _usage_from_response(response)
            raise ProviderError.from_code(
                FailureCode.PROVIDER_INCOMPLETE,
                provider_request_id=request_id,
                provider_response_id=response_id,
                response_model=getattr(response, "model", None),
                **usage,
            )
        return self._result_from_response(response)

    def _guarded_result(self, response: Any) -> TextGenerationResult:
        status = getattr(response, "status", None)
        text = getattr(response, "output_text", None)
        error = getattr(response, "error", None)
        if status == "completed" and any(
            getattr(part, "type", None) == "refusal"
            for item in (getattr(response, "output", None) or [])
            for part in (getattr(item, "content", None) or [])
        ):
            request_id, response_id = _ids_from_response(response)
            refused = ProviderError.from_code(
                FailureCode.PROVIDER_REFUSED, "Provider refused the request.",
                provider_request_id=request_id, provider_response_id=response_id,
                response_model=getattr(response, "model", None), **_usage_from_response(response),
            )
            refused.failure = refused.failure.model_copy(update={"retryability": Retryability.NO})
            raise refused
        if (
            status != "completed"
            or error is not None
            or not isinstance(text, str)
            or not text.strip()
        ):
            request_id, response_id = _ids_from_response(response)
            terminal_failure = ProviderError.from_code(
                FailureCode.PROVIDER_ERROR
                if status == "failed" or error is not None
                else FailureCode.PROVIDER_INCOMPLETE,
                provider_request_id=request_id,
                provider_response_id=response_id,
                response_model=getattr(response, "model", None),
                **_usage_from_response(response),
            )
            terminal_failure.failure = terminal_failure.failure.model_copy(
                update={"retryability": Retryability.NO}
            )
            raise terminal_failure
        return self._result_from_response(response)

    async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
        if self._guarded and (
            call.json_object
            or (call.json_schema is not None and call.max_conformance_retries != 0)
        ):
            failure = InferenceFailure.from_code(
                FailureCode.INVALID_REQUEST, "Guarded streaming does not support JSON-object mode or repair."
            )
            yield TextFailed(
                failure=failure,
                observation=_empty_failed_observation(failure.code).model_copy(
                    update={"provider_attempt_count": 0}
                ),
            )
            return
        kwargs = self._request_kwargs(call, streaming=True)
        pieces: list[str] = []
        latest_response = None
        request_id = None
        try:
            async with AsyncExitStack() as stack:
                if self._guarded:
                    # The raw official SDK stream exposes the public HTTP response,
                    # preserving its request ID even when SSE/body parsing fails.
                    raw = await self._client.responses.create(**kwargs, tools=[], stream=True)
                    response_stream = await stack.enter_async_context(raw)
                    request_id = raw.response.headers.get("x-request-id")
                else:
                    response_stream = await stack.enter_async_context(
                        self._client.responses.stream(**kwargs)
                    )
                async for event in response_stream:
                    event_type = getattr(event, "type", None)
                    response = getattr(event, "response", None)
                    if self._guarded and response is not None:
                        latest_response = response
                    if event_type == "response.output_text.delta":
                        content = getattr(event, "delta", "") or ""
                        if content:
                            if not self._guarded:
                                pieces.append(content)
                            yield TextDelta(text=content)
                    elif self._guarded and event_type in (
                        "error", "response.error", "response.refusal.delta", "response.refusal.done",
                    ):
                        code = (FailureCode.PROVIDER_REFUSED if "refusal" in event_type
                                else FailureCode.PROVIDER_ERROR)
                        raise ProviderError.from_code(code, "Provider stream did not complete successfully.")
                    elif self._guarded and event_type in (
                        "response.completed", "response.failed", "response.incomplete",
                        "response.cancelled",
                    ):
                        if event_type != "response.completed":
                            code = (FailureCode.PROVIDER_ERROR if event_type == "response.failed"
                                    else FailureCode.PROVIDER_INCOMPLETE)
                            raise ProviderError.from_code(code, "Provider stream did not complete successfully.")
                        result = self._guarded_result(response)
                        yield TextCompleted(
                            final_text=result.text,
                            observation=_completed_observation(result).model_copy(
                                update={"provider_request_id": result.provider_request_id or request_id}
                            ),
                        )
                        return
                    elif event_type == "response.error":
                        message = getattr(
                            getattr(event, "error", None), "message", "OpenAI stream error",
                        )
                        failure = InferenceFailure.from_code(FailureCode.PROVIDER_ERROR, message)
                        yield TextFailed(
                            failure=failure,
                            observation=_empty_failed_observation(failure.code),
                        )
                        return
                    elif event_type == "response.completed":
                        result = (self._result_from_response(response) if response is not None
                                  else TextGenerationResult(text="".join(pieces)))
                        yield TextCompleted(
                            final_text=result.text or "".join(pieces),
                            observation=_completed_observation(result),
                        )
                        return
            raise ProviderError.from_code(
                FailureCode.STREAM_INCOMPLETE, "OpenAI stream ended without a terminal event."
            )
        except Exception as exc:
            mapped = self._map_exception(exc)
            if self._guarded:
                # The SDK can wrap transport-gate errors in APIConnectionError.
                # Recover only our typed error; never expose arbitrary cause text.
                cause = exc
                seen = set()
                while cause is not None and id(cause) not in seen:
                    seen.add(id(cause))
                    if isinstance(cause, ProviderError):
                        mapped = cause
                        break
                    cause = cause.__cause__
            if self._guarded:
                # Header acknowledgment stays authoritative even when the body fails.
                # No response text or provider error details enter the observation.
                response_request_id, response_id = _ids_from_response(latest_response)
                yield TextFailed(
                    failure=InferenceFailure.from_code(
                        mapped.failure.code, "Guarded text stream did not complete successfully."
                    ).model_copy(
                        update={"retryability": Retryability.NO}
                    ),
                    observation=_empty_failed_observation(mapped.failure.code).model_copy(update={
                        "provider_request_id": mapped.provider_request_id or response_request_id or request_id,
                        "provider_response_id": mapped.provider_response_id or response_id,
                        "response_model": mapped.response_model or _field(latest_response, "model"),
                        **_usage_from_response(latest_response),
                        "provider_attempt_count": 1,
                    }),
                )
            else:
                yield TextFailed(
                    failure=mapped.failure,
                    observation=_empty_failed_observation(mapped.failure.code),
                )

    def _request_kwargs(
        self, call: TextGenerationCall, *, streaming: bool = False
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": call.model,
            "input": call.user_prompt,
        }
        if call.temperature is not None:
            kwargs["temperature"] = call.temperature
        if call.max_output_tokens is not None:
            kwargs["max_output_tokens"] = call.max_output_tokens
        if call.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": call.reasoning_effort}
        if call.system_prompt:
            kwargs["instructions"] = call.system_prompt
        if call.json_object:
            kwargs["text"] = {"format": {"type": "json_object"}}
        elif call.json_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": call.schema_name or "structured_output",
                    "schema": make_schema_strict(call.json_schema),
                    "strict": True,
                }
            }
        return kwargs

    def _result_from_response(self, response: Any) -> TextGenerationResult:
        request_id, response_id = _ids_from_response(response)
        if getattr(response, "refusal", None):
            raise ProviderError.from_code(
                FailureCode.PROVIDER_REFUSED,
                f"Generation refused: {response.refusal}",
                provider_request_id=request_id,
                provider_response_id=response_id,
                response_model=getattr(response, "model", None),
            )
        usage = _usage_from_response(response)
        return TextGenerationResult(
            text=getattr(response, "output_text", None),
            parsed=None,
            provider_request_id=request_id,
            provider_response_id=response_id,
            response_model=getattr(response, "model", None),
            **usage,
        )

    def _map_exception(self, exc: Exception) -> ProviderError:
        return map_openai_compatible_exception(exc)


def _empty_failed_observation(code: FailureCode) -> InferenceObservation:
    return InferenceObservation(
        provider="openai",
        latency_ms=0,
        retry_count=0,
        state=ObservationState.FAILED
        if code is not FailureCode.STREAM_INCOMPLETE
        else ObservationState.INCOMPLETE,
        failure_code=code,
    )


def _completed_observation(result: TextGenerationResult) -> InferenceObservation:
    return InferenceObservation(
        provider="openai",
        response_model=result.response_model,
        provider_request_id=result.provider_request_id,
        provider_response_id=result.provider_response_id,
        input_tokens=result.input_tokens,
        cached_input_tokens=result.cached_input_tokens,
        output_tokens=result.output_tokens,
        reasoning_tokens=result.reasoning_tokens,
        latency_ms=0,
        retry_count=0,
        state=ObservationState.COMPLETED,
    )


def _field(value: Any, name: str) -> Any:
    # Unknown SDK event variants may retain nested response data as dictionaries.
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _ids_from_response(response: Any) -> tuple[str | None, str | None]:
    return _field(response, "_request_id"), _field(response, "id")


def _usage_from_response(response: Any) -> dict[str, int | None]:
    usage = _field(response, "usage")
    input_details = _field(usage, "input_tokens_details")
    output_details = _field(usage, "output_tokens_details")
    return {
        "input_tokens": _field(usage, "input_tokens"),
        "cached_input_tokens": _field(input_details, "cached_tokens"),
        "output_tokens": _field(usage, "output_tokens"),
        "reasoning_tokens": _field(output_details, "reasoning_tokens"),
    }
