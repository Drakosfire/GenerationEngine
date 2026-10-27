"""Vercel AI Gateway Chat Completions; GE retains local schema conformance."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

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

VERCEL_BASE_URL = "https://ai-gateway.vercel.sh/v1"
PROVIDER_ID = "vercel"


class VercelTextProvider:
    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        if client is not None:
            self._client = client
            return
        require_openai_sdk(extra_name="vercel", client_cls=AsyncOpenAI)
        key = api_key if api_key is not None else os.getenv("AI_GATEWAY_API_KEY")
        if not key or not key.strip():
            raise ProviderError.from_code(
                FailureCode.CONFIGURATION_UNAVAILABLE,
                "AI_GATEWAY_API_KEY is required for Vercel text generation.",
            )
        self._client = AsyncOpenAI(api_key=key.strip(), base_url=VERCEL_BASE_URL, max_retries=0)

    async def aclose(self) -> None:
        await self._client.close()

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        kwargs = self._request_kwargs(call)
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise map_openai_compatible_exception(exc) from exc
        facts = _response_facts(response)
        choices = getattr(response, "choices", None) or []
        if len(choices) != 1 or getattr(choices[0], "index", None) != 0:
            raise _error(FailureCode.MALFORMED_PROVIDER_RESPONSE, facts)
        choice = choices[0]
        message = getattr(choice, "message", None)
        if getattr(message, "refusal", None) or choice.finish_reason == "content_filter":
            raise _error(FailureCode.PROVIDER_REFUSED, facts)
        _check_finish(choice.finish_reason, facts)
        if (
            getattr(message, "role", None) != "assistant"
            or getattr(message, "tool_calls", None)
            or getattr(message, "function_call", None)
            or not isinstance(getattr(message, "content", None), str)
        ):
            raise _error(FailureCode.MALFORMED_PROVIDER_RESPONSE, facts)
        return TextGenerationResult(text=message.content, **facts)

    async def stream(self, call: TextGenerationCall) -> AsyncIterator[TextStreamEvent]:
        stream = None
        facts: dict[str, Any] = {}
        pieces: list[str] = []
        finished = False
        try:
            kwargs = self._request_kwargs(call, streaming=True)
            stream = await self._client.chat.completions.create(**kwargs)
            try:
                async for chunk in stream:
                    update = _response_facts(chunk)
                    for key, value in update.items():
                        if value is not None:
                            facts[key] = value
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:  # The SDK may emit a final usage-only chunk.
                        continue
                    if len(choices) != 1 or getattr(choices[0], "index", None) != 0 or finished:
                        raise _error(FailureCode.MALFORMED_PROVIDER_RESPONSE, facts)
                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    reason = getattr(choice, "finish_reason", None)
                    if getattr(delta, "refusal", None) or reason == "content_filter":
                        raise _error(FailureCode.PROVIDER_REFUSED, facts)
                    if (
                        getattr(delta, "tool_calls", None)
                        or getattr(delta, "function_call", None)
                        or getattr(delta, "role", None) not in (None, "assistant")
                    ):
                        raise _error(FailureCode.MALFORMED_PROVIDER_RESPONSE, facts)
                    content = getattr(delta, "content", None)
                    if content is not None and not isinstance(content, str):
                        raise _error(FailureCode.MALFORMED_PROVIDER_RESPONSE, facts)
                    if content:
                        pieces.append(content)
                        yield TextDelta(text=content)
                    if reason is not None:
                        _check_finish(reason, facts)
                        finished = True
                if not finished:
                    raise _error(FailureCode.STREAM_INCOMPLETE, facts)
            finally:
                await stream.close()
            yield TextCompleted(final_text="".join(pieces), observation=_observation(facts))
        except Exception as exc:
            mapped = map_openai_compatible_exception(exc)
            yield TextFailed(
                failure=mapped.failure,
                observation=_observation(
                    {
                        **facts,
                        **{
                            k: getattr(mapped, k)
                            for k in _FACT_NAMES
                            if getattr(mapped, k, None) is not None
                        },
                    },
                    mapped.failure.code,
                ),
            )

    def _request_kwargs(
        self, call: TextGenerationCall, *, streaming: bool = False
    ) -> dict[str, Any]:
        if call.provider_route is not None:
            raise ProviderError.from_code(
                FailureCode.UNSUPPORTED_CAPABILITY,
                "OpenRouter provider_route is not supported by Vercel text generation.",
            )
        if call.reasoning_effort not in (
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            raise ProviderError.from_code(
                FailureCode.UNSUPPORTED_CAPABILITY, "Unsupported reasoning effort."
            )
        messages = []
        if call.system_prompt:
            messages.append({"role": "system", "content": call.system_prompt})
        messages.append({"role": "user", "content": call.user_prompt})
        kwargs: dict[str, Any] = {"model": call.model, "messages": messages}
        if call.temperature is not None:
            kwargs["temperature"] = call.temperature
        if call.max_output_tokens is not None:
            kwargs["max_tokens"] = call.max_output_tokens
        if call.reasoning_effort is not None:
            kwargs["reasoning_effort"] = call.reasoning_effort
        if call.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": call.schema_name or "response",
                    "schema": call.json_schema,
                    "strict": True,
                },
            }
        elif call.json_object:
            kwargs["response_format"] = {"type": "json"}
        if streaming:
            kwargs.update(stream=True, stream_options={"include_usage": True})
        return kwargs


_FACT_NAMES = (
    "provider_request_id",
    "provider_response_id",
    "response_model",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
)


def _response_facts(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "provider_request_id": getattr(response, "_request_id", None),
        "provider_response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "cached_input_tokens": getattr(prompt_details, "cached_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
    }


def _error(code: FailureCode, facts: dict[str, Any]) -> ProviderError:
    return ProviderError.from_code(
        code, "Vercel text response did not complete successfully.", **facts
    )


def _check_finish(reason: str | None, facts: dict[str, Any]) -> None:
    if reason == "length":
        raise _error(FailureCode.PROVIDER_INCOMPLETE, facts)
    if reason == "content_filter":
        raise _error(FailureCode.PROVIDER_REFUSED, facts)
    if reason != "stop":
        raise _error(FailureCode.MALFORMED_PROVIDER_RESPONSE, facts)


def _observation(facts: dict[str, Any], failure: FailureCode | None = None) -> InferenceObservation:
    state = ObservationState.COMPLETED
    if failure is not None:
        state = (
            ObservationState.INCOMPLETE
            if failure == FailureCode.STREAM_INCOMPLETE
            else ObservationState.FAILED
        )
    return InferenceObservation(
        provider=PROVIDER_ID,
        **facts,
        latency_ms=0,
        retry_count=0,
        state=state,
        failure_code=failure,
    )
