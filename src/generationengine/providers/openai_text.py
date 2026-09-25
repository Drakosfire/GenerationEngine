"""OpenAI text/structured/stream adapter. SDK types stay inside this module."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from generationengine.failures import FailureCode, InferenceFailure
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
from generationengine.utils.schema_utils import make_schema_strict


class OpenAITextProvider:
    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
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
        self._client = AsyncOpenAI(api_key=key, max_retries=0)

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        kwargs = self._request_kwargs(call)
        try:
            response = await self._client.responses.create(**kwargs)
        except Exception as exc:
            raise self._map_exception(exc) from exc
        return self._result_from_response(response)

    async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
        kwargs = self._request_kwargs(call, streaming=True)
        pieces: list[str] = []
        try:
            stream_manager = self._client.responses.stream(**kwargs)
            async with stream_manager as response_stream:
                async for event in response_stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        content = getattr(event, "delta", "") or ""
                        if content:
                            pieces.append(content)
                            yield TextDelta(text=content)
                    elif event_type == "response.error":
                        message = getattr(
                            getattr(event, "error", None),
                            "message",
                            "OpenAI stream error",
                        )
                        failure = InferenceFailure.from_code(
                            FailureCode.PROVIDER_ERROR,
                            message,
                        )
                        yield TextFailed(
                            failure=failure,
                            observation=_empty_failed_observation(failure.code),
                        )
                        return
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        result = (
                            self._result_from_response(response)
                            if response is not None
                            else TextGenerationResult(text="".join(pieces))
                        )
                        yield TextCompleted(
                            final_text=result.text or "".join(pieces),
                            observation=_completed_observation(result),
                        )
                        return
            failure = InferenceFailure.from_code(
                FailureCode.STREAM_INCOMPLETE,
                "OpenAI stream ended without a terminal event.",
            )
            yield TextFailed(
                failure=failure,
                observation=_empty_failed_observation(failure.code),
            )
        except ProviderError as exc:
            yield TextFailed(
                failure=exc.failure,
                observation=_empty_failed_observation(exc.failure.code),
            )
        except Exception as exc:
            mapped = self._map_exception(exc)
            yield TextFailed(
                failure=mapped.failure,
                observation=_empty_failed_observation(mapped.failure.code),
            )

    def _request_kwargs(self, call: TextGenerationCall, *, streaming: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": call.model,
            "input": call.user_prompt,
        }
        if call.temperature is not None:
            kwargs["temperature"] = call.temperature
        if call.max_output_tokens is not None:
            kwargs["max_output_tokens"] = call.max_output_tokens
        if call.system_prompt:
            kwargs["instructions"] = call.system_prompt
        if call.json_object:
            kwargs["text"] = {"format": {"type": "json_object"}}
        elif call.json_schema and not streaming:
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
        text = getattr(response, "output_text", None)
        usage = getattr(response, "usage", None)
        cached = None
        if usage is not None:
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details is not None:
                cached = getattr(input_details, "cached_tokens", None)
        return TextGenerationResult(
            text=text,
            parsed=None,
            provider_request_id=request_id,
            provider_response_id=response_id,
            response_model=getattr(response, "model", None),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            cached_input_tokens=cached,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
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
        latency_ms=0,
        retry_count=0,
        state=ObservationState.COMPLETED,
    )


def _ids_from_response(response: Any) -> tuple[str | None, str | None]:
    return getattr(response, "_request_id", None), getattr(response, "id", None)
