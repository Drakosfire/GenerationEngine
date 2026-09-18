"""OpenRouter text adapter. Provider identity is openrouter, not openai.

Structured generation uses ordinary chat completions plus JSON instructions.
Provider-native json_schema is not sent. GenerationEngine local validation is
the structured contract.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from generationengine.conformance import schema_instruction
from generationengine.failures import FailureCode
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_ID = "openrouter"


class OpenRouterTextProvider:
    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        if client is not None:
            self._client = client
            return
        require_openai_sdk(extra_name="openrouter", client_cls=AsyncOpenAI)
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ProviderError.from_code(
                FailureCode.CONFIGURATION_UNAVAILABLE,
                "OPENROUTER_API_KEY is required for OpenRouter text generation.",
            )
        self._client = AsyncOpenAI(
            api_key=key,
            base_url=OPENROUTER_BASE_URL,
            max_retries=0,
        )

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        kwargs = self._request_kwargs(call)
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise self._map_exception(exc) from exc
        return self._result_from_response(response)

    async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
        pieces: list[str] = []
        usage = None
        request_id = None
        response_id = None
        response_model = None
        try:
            kwargs = self._request_kwargs(call, streaming=True)
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                request_id = getattr(chunk, "_request_id", None) or request_id
                response_id = getattr(chunk, "id", None) or response_id
                response_model = getattr(chunk, "model", None) or response_model
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) or ""
                if content:
                    pieces.append(content)
                    yield TextDelta(text=content)
            result = TextGenerationResult(
                text="".join(pieces),
                provider_request_id=request_id,
                provider_response_id=response_id,
                response_model=response_model,
                **_usage_fields(usage),
            )
            yield TextCompleted(
                final_text=result.text or "",
                observation=_completed_observation(result),
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
        messages: list[dict[str, str]] = []
        system_prompt = call.system_prompt
        if call.json_schema and not streaming:
            guide = schema_instruction(call.json_schema)
            system_prompt = f"{system_prompt}\n\n{guide}" if system_prompt else guide
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": call.user_prompt})
        kwargs: dict[str, Any] = {
            "model": call.model,
            "messages": messages,
        }
        if call.temperature is not None:
            kwargs["temperature"] = call.temperature
        if streaming:
            kwargs["stream"] = True
        return kwargs

    def _result_from_response(self, response: Any) -> TextGenerationResult:
        request_id, response_id = _ids_from_response(response)
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        refusal = getattr(message, "refusal", None) if message is not None else None
        if refusal:
            raise ProviderError.from_code(
                FailureCode.PROVIDER_REFUSED,
                f"Generation refused: {refusal}",
                provider_request_id=request_id,
                provider_response_id=response_id,
                response_model=getattr(response, "model", None),
            )
        text = getattr(message, "content", None) if message is not None else None
        usage = getattr(response, "usage", None)
        return TextGenerationResult(
            text=text,
            parsed=None,
            provider_request_id=request_id,
            provider_response_id=response_id,
            response_model=getattr(response, "model", None),
            **_usage_fields(usage),
        )

    def _map_exception(self, exc: Exception) -> ProviderError:
        return map_openai_compatible_exception(exc)


def _usage_fields(usage: Any) -> dict[str, int | None]:
    if usage is None:
        return {
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
        }
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "cached_input_tokens": cached,
        "output_tokens": getattr(usage, "completion_tokens", None),
    }


def _empty_failed_observation(code: FailureCode) -> InferenceObservation:
    return InferenceObservation(
        provider=PROVIDER_ID,
        latency_ms=0,
        retry_count=0,
        state=ObservationState.FAILED
        if code is not FailureCode.STREAM_INCOMPLETE
        else ObservationState.INCOMPLETE,
        failure_code=code,
    )


def _completed_observation(result: TextGenerationResult) -> InferenceObservation:
    return InferenceObservation(
        provider=PROVIDER_ID,
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
