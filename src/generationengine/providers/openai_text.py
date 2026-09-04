"""OpenAI text/structured/stream adapter. SDK types stay inside this module."""

from __future__ import annotations

import json
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
from generationengine.utils.schema_utils import make_schema_strict

try:
    from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
except ImportError:
    AsyncOpenAI = None  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIError = Exception  # type: ignore


def _require_openai() -> None:
    if AsyncOpenAI is None:
        raise ProviderError.from_code(
            FailureCode.CONFIGURATION_UNAVAILABLE,
            "openai extra is not installed. Install generationengine[openai].",
        )


class OpenAITextProvider:
    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        if client is not None:
            self._client = client
            return
        _require_openai()
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderError.from_code(
                FailureCode.CONFIGURATION_UNAVAILABLE,
                "OPENAI_API_KEY is required for text generation.",
            )
        self._client = AsyncOpenAI(api_key=key)

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        kwargs = self._request_kwargs(call)
        try:
            response = await self._client.responses.create(**kwargs)
        except Exception as exc:
            raise self._map_exception(exc) from exc
        return self._result_from_response(response, structured=call.json_schema is not None)

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
                            self._result_from_response(response, structured=False)
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
            "temperature": call.temperature,
        }
        if call.system_prompt:
            kwargs["instructions"] = call.system_prompt
        if call.json_schema and not streaming:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": call.schema_name or "structured_output",
                    "schema": make_schema_strict(call.json_schema),
                    "strict": True,
                }
            }
        return kwargs

    def _result_from_response(self, response: Any, *, structured: bool) -> TextGenerationResult:
        if getattr(response, "refusal", None):
            raise ProviderError.from_code(
                FailureCode.PROVIDER_REFUSED,
                f"Generation refused: {response.refusal}",
                provider_request_id=getattr(response, "id", None),
                response_model=getattr(response, "model", None),
            )
        text = getattr(response, "output_text", None)
        usage = getattr(response, "usage", None)
        parsed = None
        if structured and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProviderError.from_code(
                    FailureCode.STRUCTURED_OUTPUT_INVALID,
                    f"Structured output was not valid JSON: {exc}",
                    provider_request_id=getattr(response, "id", None),
                    response_model=getattr(response, "model", None),
                ) from exc
        cached = None
        if usage is not None:
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details is not None:
                cached = getattr(input_details, "cached_tokens", None)
        return TextGenerationResult(
            text=text,
            parsed=parsed,
            provider_request_id=getattr(response, "id", None),
            response_model=getattr(response, "model", None),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            cached_input_tokens=cached,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
        )

    def _map_exception(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        name = type(exc).__name__
        message = str(exc) or name
        if isinstance(exc, RateLimitError) or "RateLimit" in name:
            return ProviderError.from_code(FailureCode.RATE_LIMITED, message)
        if isinstance(exc, APITimeoutError) or "Timeout" in name:
            return ProviderError.from_code(FailureCode.PROVIDER_TIMEOUT, message)
        status = getattr(exc, "status_code", None)
        if status == 429:
            return ProviderError.from_code(FailureCode.RATE_LIMITED, message)
        if isinstance(status, int) and status >= 500:
            return ProviderError.from_code(FailureCode.PROVIDER_UNAVAILABLE, message)
        if isinstance(exc, APIError):
            return ProviderError.from_code(FailureCode.PROVIDER_ERROR, message)
        return ProviderError.from_code(FailureCode.PROVIDER_ERROR, message)


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
        input_tokens=result.input_tokens,
        cached_input_tokens=result.cached_input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=0,
        retry_count=0,
        state=ObservationState.COMPLETED,
    )
