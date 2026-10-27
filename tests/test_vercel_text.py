"""Real OpenAI SDK / mock HTTP tests for the fixed Vercel text transport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from generationengine import GenerationClient, TextRequest
from generationengine.failures import FailureCode
from generationengine.providers.base import TextCompleted, TextDelta, TextFailed, TextGenerationCall
from generationengine.providers.errors import ProviderError
from generationengine.providers.vercel_text import VERCEL_BASE_URL, VercelTextProvider
from generationengine.types import GenerationEngineError, ProviderRoute

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def completion(text='{"name":"oak"}', *, reason="stop", usage=True, refusal=None):
    value = {
        "id": "gateway-response",
        "object": "chat.completion",
        "created": 1,
        "model": "vendor/resolved-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "refusal": refusal},
                "finish_reason": reason,
            }
        ],
    }
    if usage:
        value["usage"] = {
            "prompt_tokens": 31,
            "completion_tokens": 7,
            "total_tokens": 38,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
    return value


def engine_with(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sdk = AsyncOpenAI(api_key="mock-key", base_url=VERCEL_BASE_URL, max_retries=0, http_client=http)
    provider = VercelTextProvider(client=sdk)
    return GenerationClient(text_providers={"vercel": provider}), provider, http


@pytest.mark.asyncio
async def test_real_sdk_wire_explicit_uncatalogued_target_and_local_schema():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion(), headers={"x-request-id": "http-id"})

    engine, _, http = engine_with(handler)
    try:
        result = await engine.generate_structured(
            TextRequest(
                provider="vercel",
                model="vendor/not-in-catalog",
                system_prompt="Exact instructions",
                user_prompt="Exact source",
                temperature=None,
                reasoning_effort="low",
                max_output_tokens=1234,
                max_transport_retries=0,
                json_schema=SCHEMA,
                schema_name="Tree",
                max_conformance_retries=0,
            )
        )
        assert result.parsed == {"name": "oak"}
        obs = result.observation
        assert obs.provider == "vercel"
        assert obs.requested_model == obs.resolved_model == "vendor/not-in-catalog"
        assert obs.response_model == "vendor/resolved-model"
        assert obs.provider_request_id == "http-id"
        assert obs.provider_response_id == "gateway-response"
        assert (
            obs.input_tokens,
            obs.cached_input_tokens,
            obs.output_tokens,
            obs.reasoning_tokens,
        ) == (31, 0, 7, 2)
        assert len(requests) == 1
        request = requests[0]
        assert str(request.url) == VERCEL_BASE_URL + "/chat/completions"
        body = json.loads(request.content)
        assert body == {
            "model": "vendor/not-in-catalog",
            "messages": [
                {"role": "system", "content": "Exact instructions"},
                {"role": "user", "content": "Exact source"},
            ],
            "reasoning_effort": "low",
            "max_tokens": 1234,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Tree", "schema": SCHEMA, "strict": True},
            },
        }
    finally:
        await engine.aclose()
    assert http.is_closed


@pytest.mark.asyncio
async def test_local_conformance_rejects_provider_claim_of_schema_success():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion('{"name": 4}'))

    engine, _, _ = engine_with(handler)
    try:
        with pytest.raises(GenerationEngineError) as exc:
            await engine.generate_structured(
                TextRequest(
                    provider="vercel",
                    model="vendor/model",
                    user_prompt="source",
                    json_schema=SCHEMA,
                    max_conformance_retries=0,
                    max_transport_retries=0,
                )
            )
        assert exc.value.failure.code is FailureCode.STRUCTURED_OUTPUT_INVALID
        assert len(calls) == 1
    finally:
        await engine.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,refusal,code",
    [
        ("length", None, FailureCode.PROVIDER_INCOMPLETE),
        ("content_filter", None, FailureCode.PROVIDER_REFUSED),
        ("stop", "private refusal body", FailureCode.PROVIDER_REFUSED),
        ("tool_calls", None, FailureCode.MALFORMED_PROVIDER_RESPONSE),
        (None, None, FailureCode.MALFORMED_PROVIDER_RESPONSE),
    ],
)
async def test_terminal_failure_not_usable_text_and_usage_retained(reason, refusal, code):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json=completion(reason=reason, refusal=refusal),
            headers={"x-request-id": "http-id"},
        )

    engine, _, _ = engine_with(handler)
    try:
        with pytest.raises(GenerationEngineError) as exc:
            await engine.generate_text(
                TextRequest(
                    provider="vercel",
                    model="vendor/model",
                    user_prompt="source",
                    max_transport_retries=0,
                )
            )
        assert exc.value.failure.code is code
        assert exc.value.observation.input_tokens == 31
        assert exc.value.observation.provider_request_id == "http-id"
        assert "private refusal body" not in str(exc.value)
        assert len(calls) == 1
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_unknown_usage_stays_unknown_and_json_object_controls():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion(usage=False))

    engine, _, _ = engine_with(handler)
    try:
        result = await engine.generate_text(
            TextRequest(
                provider="vercel",
                model="vendor/model",
                user_prompt="source",
                temperature=0,
                json_object=True,
            )
        )
        assert result.parsed is None
        assert result.observation.input_tokens is None
        assert result.observation.cached_input_tokens is None
        assert result.observation.reasoning_tokens is None
        assert bodies[0]["temperature"] == 0
        assert bodies[0]["response_format"] == {"type": "json"}
        assert "max_tokens" not in bodies[0]
        assert "reasoning_effort" not in bodies[0]
    finally:
        await engine.aclose()


def test_lazy_registration_and_missing_credential_without_io(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    engine = GenerationClient.from_env()
    assert engine._text_providers == {}
    with pytest.raises(GenerationEngineError) as exc:
        engine._text_provider_for("vercel")
    assert exc.value.failure.code is FailureCode.CONFIGURATION_UNAVAILABLE
    assert exc.value.observation.provider == "vercel"


def test_constructor_fixed_endpoint_and_sdk_retries_disabled(monkeypatch):
    captured = {}
    monkeypatch.setenv("AI_GATEWAY_API_KEY", " gateway-key ")
    monkeypatch.setattr(
        "generationengine.providers.vercel_text.AsyncOpenAI",
        lambda **kwargs: captured.update(kwargs),
    )
    VercelTextProvider()
    assert captured == {"api_key": "gateway-key", "base_url": VERCEL_BASE_URL, "max_retries": 0}


@pytest.mark.asyncio
async def test_unsupported_route_and_effort_fail_before_dispatch():
    def no_io(request):
        raise AssertionError("unexpected provider dispatch")

    _, provider, http = engine_with(no_io)
    try:
        for kwargs in [
            {"provider_route": ProviderRoute(order=("one",), allow_fallbacks=False)},
            {"reasoning_effort": "invented"},
        ]:
            with pytest.raises(ProviderError) as exc:
                await provider.generate(
                    TextGenerationCall(model="vendor/model", user_prompt="x", **kwargs)
                )
            assert exc.value.failure.code is FailureCode.UNSUPPORTED_CAPABILITY
    finally:
        await provider.aclose()
    assert http.is_closed


def chunk(content=None, reason=None, *, refusal=None):
    return {
        "id": "stream-response",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "vendor/resolved-model",
        "choices": [
            {"index": 0, "delta": {"content": content, "refusal": refusal}, "finish_reason": reason}
        ],
    }


class SSE(httpx.AsyncByteStream):
    def __init__(self, events, *, stall=False):
        self.events = events
        self.closed = False
        self.stall = stall

    async def __aiter__(self):
        for event in self.events:
            yield ("data: " + json.dumps(event) + "\n\n").encode()
        if self.stall:
            await asyncio.Event().wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,code",
    [
        ("stop", None),
        ("length", FailureCode.PROVIDER_INCOMPLETE),
        (None, FailureCode.STREAM_INCOMPLETE),
        ("content_filter", FailureCode.PROVIDER_REFUSED),
    ],
)
async def test_real_sse_terminal_truth_and_close(reason, code):
    usage = completion()["usage"]
    events = [chunk('{"name":"oak"}', reason)]
    events.append(
        {
            "id": "stream-response",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "vendor/resolved-model",
            "choices": [],
            "usage": usage,
        }
    )
    wire = SSE(events)
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, stream=wire, headers={"content-type": "text/event-stream"})

    engine, _, http = engine_with(handler)
    try:
        seen = [
            event
            async for event in engine.stream_text(
                TextRequest(
                    provider="vercel",
                    model="vendor/model",
                    user_prompt="source",
                    temperature=None,
                    max_conformance_retries=0,
                    json_schema=SCHEMA,
                )
            )
        ]
        assert wire.closed
        assert bodies[0]["stream_options"] == {"include_usage": True}
        assert bodies[0]["response_format"]["json_schema"]["schema"] == SCHEMA
        if reason != "content_filter":
            assert isinstance(seen[0], TextDelta)
        if code:
            assert isinstance(seen[-1], TextFailed)
            assert seen[-1].failure.code is code
            assert not any(isinstance(e, TextCompleted) for e in seen)
        else:
            assert isinstance(seen[-1], TextCompleted)
            assert seen[-1].final_text == '{"name":"oak"}'
            assert seen[-1].observation.input_tokens == 31
    finally:
        await engine.aclose()
    assert http.is_closed


@pytest.mark.asyncio
async def test_early_stream_close_closes_response():
    wire = SSE([chunk("partial")], stall=True)
    engine, provider, _ = engine_with(
        lambda request: httpx.Response(
            200, stream=wire, headers={"content-type": "text/event-stream"}
        )
    )
    stream = provider.stream(TextGenerationCall(model="vendor/model", user_prompt="x"))
    assert isinstance(await anext(stream), TextDelta)
    await stream.aclose()
    assert wire.closed
    await engine.aclose()


@pytest.mark.asyncio
async def test_transport_deadline_and_error_are_owned_by_ge():
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.sleep(1)
        return httpx.Response(200, json=completion())

    engine, _, _ = engine_with(handler)
    try:
        with pytest.raises(GenerationEngineError) as exc:
            await engine.generate_text(
                TextRequest(
                    provider="vercel",
                    model="vendor/model",
                    user_prompt="x",
                    deadline_ms=10,
                    max_transport_retries=0,
                )
            )
        assert exc.value.failure.code is FailureCode.PROVIDER_TIMEOUT
        assert len(calls) == 1
    finally:
        await engine.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["empty", "multiple", "wrong_index", "tools", "wrong_role"])
async def test_malformed_choices_never_deliver(shape):
    response = completion()
    if shape == "empty":
        response["choices"] = []
    elif shape == "multiple":
        response["choices"].append({**response["choices"][0], "index": 1})
    elif shape == "wrong_index":
        response["choices"][0]["index"] = 2
    elif shape == "tools":
        response["choices"][0]["message"]["tool_calls"] = [
            {"id": "tool", "type": "function", "function": {"name": "x", "arguments": "{}"}}
        ]
    else:
        response["choices"][0]["message"]["role"] = "user"
    engine, _, _ = engine_with(lambda request: httpx.Response(200, json=response))
    try:
        with pytest.raises(GenerationEngineError) as exc:
            await engine.generate_text(
                TextRequest(provider="vercel", model="vendor/model", user_prompt="x")
            )
        assert exc.value.failure.code is FailureCode.MALFORMED_PROVIDER_RESPONSE
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_structural_retry_remains_ge_owned():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json=completion('{"name":3}' if len(bodies) == 1 else '{"name":"oak"}')
        )

    engine, _, _ = engine_with(handler)
    try:
        result = await engine.generate_structured(
            TextRequest(
                provider="vercel",
                model="vendor/model",
                user_prompt="x",
                json_schema=SCHEMA,
                reasoning_effort="none",
                max_transport_retries=0,
            )
        )
        assert result.parsed == {"name": "oak"}
        assert len(bodies) == 2
        assert result.observation.conformance_retry_count == 1
        assert result.observation.provider_attempt_count == 2
        assert all(b["reasoning_effort"] == "none" for b in bodies)
        assert all(b["model"] == "vendor/model" for b in bodies)
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_transport_retry_count_is_not_sdk_retry():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                503, json={"error": {"message": "private body", "type": "server_error"}}
            )
        return httpx.Response(200, json=completion())

    engine, _, _ = engine_with(handler)
    try:
        result = await engine.generate_text(
            TextRequest(
                provider="vercel", model="vendor/model", user_prompt="x", max_transport_retries=1
            )
        )
        assert len(calls) == 2
        assert result.observation.retry_count == 1
        assert result.observation.provider_attempt_count == 2
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_duplicate_stream_terminal_is_failure():
    wire = SSE([chunk("one", "stop"), chunk(None, "stop")])
    engine, _, _ = engine_with(
        lambda request: httpx.Response(
            200, stream=wire, headers={"content-type": "text/event-stream"}
        )
    )
    try:
        seen = [
            event
            async for event in engine.stream_text(
                TextRequest(provider="vercel", model="vendor/model", user_prompt="x")
            )
        ]
        assert wire.closed
        assert isinstance(seen[-1], TextFailed)
        assert seen[-1].failure.code is FailureCode.MALFORMED_PROVIDER_RESPONSE
        assert not any(isinstance(e, TextCompleted) for e in seen)
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_stream_cancellation_closes_response():
    wire = SSE([chunk("partial")], stall=True)
    engine, provider, _ = engine_with(
        lambda request: httpx.Response(
            200, stream=wire, headers={"content-type": "text/event-stream"}
        )
    )
    stream = provider.stream(TextGenerationCall(model="vendor/model", user_prompt="x"))
    assert isinstance(await anext(stream), TextDelta)
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert wire.closed
    await engine.aclose()
