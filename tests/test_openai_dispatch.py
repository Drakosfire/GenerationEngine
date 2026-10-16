"""Real installed OpenAI SDK serialization, synthetic HTTP only."""

import asyncio
import hashlib
import json
from dataclasses import FrozenInstanceError

import httpx
import pytest

from generationengine import GenerationClient, GenerationEngineError, TextRequest
from generationengine.providers.base import TextGenerationCall
from generationengine.providers.errors import ProviderError
from generationengine.providers.openai_dispatch import GuardedOpenAITransport
from generationengine.providers.openai_text import OpenAITextProvider
from generationengine.text_dispatch import TextDispatchGate

ENDPOINT = "https://api.openai.com/v1/responses"
SECRET = "synthetic-credential-not-for-observer"


def request(**changes):
    return TextRequest(
        provider="openai",
        model="writer-test",
        user_prompt='Néra says "yes"\n雪',
        system_prompt="Use only supplied evidence.",
        temperature=None,
        max_output_tokens=1024,
        reasoning_effort="low",
        max_transport_retries=3,
        **changes,
    )


class Wire(httpx.AsyncBaseTransport):
    def __init__(self, outcome=None):
        self.calls = []
        self.closed = False
        self.outcome = outcome
        self.response = None

    async def handle_async_request(self, req):
        self.calls.append(req)
        await asyncio.sleep(0)
        if self.outcome == "timeout":
            raise httpx.ReadTimeout(SECRET, request=req)
        if self.outcome == "cancel":
            raise asyncio.CancelledError()
        status = self.outcome if isinstance(self.outcome, int) else 200
        content = (
            b"not JSON"
            if self.outcome == "malformed"
            else json.dumps(
                {
                    "id": "resp_synthetic",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "writer-test",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_synthetic",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Supported prose [S1].",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                }
            ).encode()
        )
        self.response = httpx.Response(
            status,
            content=content,
            headers={"content-type": "application/json", "x-request-id": "req_synthetic"},
            request=req,
        )
        return self.response

    async def aclose(self):
        self.closed = True


def client(wire, authorize, record):
    provider = OpenAITextProvider(
        api_key=SECRET,
        dispatch_gate=TextDispatchGate(authorize, record),
        transport=wire,
    )
    return GenerationClient(text_providers={"openai": provider})


@pytest.mark.asyncio
async def test_actual_sdk_exact_bytes_identity_usage_and_no_secret_headers(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/ignored")
    wire, views, events = Wire(), [], []

    async def authorize(view):
        views.append(view)
        with pytest.raises(FrozenInstanceError):
            view.model = "mutated"
        return True

    async def record(event):
        events.append(event)
        return True

    engine = client(wire, authorize, record)
    try:
        result = await engine.generate_text(request(json_object=True))
    finally:
        await engine.aclose()
    assert result.text == "Supported prose [S1]."
    assert result.observation.input_tokens == 20
    assert result.observation.output_tokens == 8
    assert result.observation.provider_request_id == "req_synthetic"
    assert result.observation.provider_response_id == "resp_synthetic"
    assert len(views) == len(wire.calls) == 1
    view, sent = views[0], wire.calls[0]
    assert (view.provider, view.api_mode, view.method, view.url) == (
        "openai",
        "responses",
        "POST",
        ENDPOINT,
    )
    assert view.body == sent.content
    assert view.body_sha256 == hashlib.sha256(sent.content).hexdigest()
    assert view.body_size == len(sent.content)
    assert json.loads(sent.content) == {
        "model": "writer-test",
        "input": request().user_prompt,
        "instructions": request().system_prompt,
        "max_output_tokens": 1024,
        "reasoning": {"effort": "low"},
        "text": {"format": {"type": "json_object"}},
        "tools": [],
        "stream": False,
    }
    assert sent.headers["authorization"] == f"Bearer {SECRET}"
    assert SECRET not in repr(view) and "headers" not in vars(view)
    assert [e.outcome for e in events] == ["sending", "response_received"]
    assert events[-1].http_status == 200 and wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["deny", "truthy", "raise", "cancel", "record_fail", "record_raise", "record_cancel"]
)
async def test_before_send_failure_never_sends(action):
    wire, events = Wire(), []

    async def authorize(_):
        if action == "raise":
            raise RuntimeError(SECRET)
        if action == "cancel":
            raise asyncio.CancelledError()
        return 1 if action == "truthy" else action != "deny"

    async def record(event):
        events.append(event)
        if event.outcome == "sending":
            if action == "record_raise":
                raise RuntimeError(SECRET)
            if action == "record_cancel":
                raise asyncio.CancelledError()
            return action != "record_fail"
        return True

    engine = client(wire, authorize, record)
    try:
        with pytest.raises(
            asyncio.CancelledError if "cancel" in action else GenerationEngineError
        ) as error:
            await engine.generate_text(request())
        if "cancel" not in action:
            assert SECRET not in str(error.value)
            assert error.value.observation.provider_attempt_count == 1
    finally:
        await engine.aclose()
    assert not wire.calls and events[-1].outcome == "known_not_sent" and wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel", "malformed", 429, 503, 307])
async def test_transport_failure_or_response_never_retries_or_redirects(outcome):
    wire, events = Wire(outcome), []

    async def authorize(_):
        return True

    async def record(event):
        events.append(event)
        return True

    engine = client(wire, authorize, record)
    try:
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancel" else GenerationEngineError
        ) as error:
            await engine.generate_text(request())
        if outcome != "cancel":
            assert error.value.observation.provider_attempt_count == 1
            assert error.value.observation.retry_count == 0
    finally:
        await engine.aclose()
    assert len(wire.calls) == 1
    assert [e.outcome for e in events] == [
        "sending",
        "outcome_unknown" if outcome in ("timeout", "cancel") else "response_received",
    ]
    assert wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["authorize", "sending"])
@pytest.mark.parametrize("mutation", ["method", "url", "stream", "headers"])
async def test_callbacks_cannot_mutate_prepared_request(phase, mutation):
    wire, events = Wire(), []
    req = httpx.Request(
        "POST", ENDPOINT, json={"model": "writer-test", "tools": [], "stream": False}
    )

    def mutate():
        if mutation == "method":
            req.method = "GET"
        elif mutation == "url":
            req.url = httpx.URL("https://example.invalid/elsewhere")
        elif mutation == "stream":
            req.stream = httpx.ByteStream(b"changed")
        else:
            req.headers["authorization"] = "changed"

    async def authorize(_):
        if phase == "authorize":
            mutate()
        return True

    async def record(event):
        events.append(event)
        if phase == "sending" and event.outcome == "sending":
            mutate()
        return True

    transport = GuardedOpenAITransport(TextDispatchGate(authorize, record), wire)
    try:
        with pytest.raises(ProviderError):
            await transport.handle_async_request(req)
    finally:
        await transport.aclose()
    assert not wire.calls and events[-1].outcome == "known_not_sent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["endpoint", "query", "get", "tools", "stream", "compression", "model", "scalar"]
)
async def test_unsupported_request_refuses_before_authorization(change):
    wire = Wire()
    body = {"model": "writer-test", "tools": [], "stream": False}
    url, method, headers = ENDPOINT, "POST", {}
    if change == "endpoint":
        url = "https://example.invalid/v1/responses"
    elif change == "query":
        url += "?different=true"
    elif change == "get":
        method = "GET"
    elif change == "tools":
        body["tools"] = [{"type": "web_search"}]
    elif change == "stream":
        body["stream"] = True
    elif change == "compression":
        headers["content-encoding"] = "gzip"
    elif change == "model":
        body["model"] = ""
    else:
        body = None

    async def unexpected(_):
        pytest.fail("Unsupported request reached authorization")

    transport = GuardedOpenAITransport(TextDispatchGate(unexpected, unexpected), wire)
    try:
        with pytest.raises(ProviderError):
            await transport.handle_async_request(
                httpx.Request(method, url, json=body, headers=headers)
            )
    finally:
        await transport.aclose()
    assert not wire.calls


@pytest.mark.asyncio
async def test_response_ack_failure_closes_response_without_replay():
    wire, events = Wire(), []

    async def authorize(_):
        return True

    async def record(event):
        events.append(event)
        return event.outcome != "response_received"

    engine = client(wire, authorize, record)
    try:
        with pytest.raises(GenerationEngineError):
            await engine.generate_text(request())
    finally:
        await engine.aclose()
    assert len(wire.calls) == 1 and wire.response.is_closed
    assert [e.outcome for e in events] == ["sending", "response_received"]


@pytest.mark.asyncio
async def test_concurrent_requests_have_independent_ids_and_owned_cleanup():
    wire, views, events = Wire(), [], []

    async def authorize(view):
        views.append(view)
        await asyncio.sleep(0)
        return True

    async def record(event):
        events.append(event)
        return True

    engine = client(wire, authorize, record)
    try:
        await asyncio.gather(*(engine.generate_text(request()) for _ in range(3)))
    finally:
        await engine.aclose()
    assert len(wire.calls) == len({v.attempt_id for v in views}) == 3
    for view in views:
        assert [e.outcome for e in events if e.request == view] == ["sending", "response_received"]
    assert wire.closed


@pytest.mark.asyncio
async def test_guarded_stream_and_structured_repair_refuse_without_io():
    wire = Wire()

    async def unexpected(_):
        pytest.fail("Unsupported capability reached gate")

    provider = OpenAITextProvider(
        api_key=SECRET, dispatch_gate=TextDispatchGate(unexpected, unexpected), transport=wire
    )
    try:
        events = [
            e
            async for e in provider.stream(TextGenerationCall(model="writer-test", user_prompt="x"))
        ]
        assert len(events) == 1 and events[0].failure.code.value == "INVALID_REQUEST"
        with pytest.raises(ProviderError):
            await provider.generate(
                TextGenerationCall(
                    model="writer-test", user_prompt="x", json_schema={"type": "object"}
                )
            )
    finally:
        await provider.aclose()
    assert not wire.calls


def test_injected_client_cannot_bypass_gate():
    async def acknowledge(_):
        return True

    with pytest.raises(ValueError):
        OpenAITextProvider(
            client=object(), dispatch_gate=TextDispatchGate(acknowledge, acknowledge)
        )
    with pytest.raises(ValueError):
        OpenAITextProvider(api_key=SECRET, transport=Wire())


@pytest.mark.asyncio
async def test_body_failure_preserves_response_headers_acknowledgment():
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"output":'
            raise httpx.ReadTimeout("synthetic incomplete response")

    class BodyWire(Wire):
        async def handle_async_request(self, req):
            self.calls.append(req)
            self.response = httpx.Response(200, stream=BrokenBody(), request=req)
            return self.response

    wire, events = BodyWire(), []

    async def authorize(_):
        return True

    async def record(event):
        events.append(event)
        return True

    engine = client(wire, authorize, record)
    try:
        with pytest.raises(GenerationEngineError) as error:
            await engine.generate_text(request())
        assert error.value.observation.provider_attempt_count == 1
    finally:
        await engine.aclose()
    assert len(wire.calls) == 1 and wire.response.is_closed
    assert [e.outcome for e in events] == ["sending", "response_received"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        "failed",
        "cancelled",
        "queued",
        "in_progress",
        "incomplete",
        "unknown",
        None,
        "completed_with_error",
        "empty",
        "blank",
    ],
)
async def test_residual_text_does_not_make_unsuccessful_response_successful(status):
    class StatusWire(Wire):
        async def handle_async_request(self, req):
            response = await super().handle_async_request(req)
            data = json.loads(response.content)
            if status in {"empty", "blank"}:
                data["output"][0]["content"][0]["text"] = "" if status == "empty" else "  \n"
            else:
                data["status"] = "completed" if status == "completed_with_error" else status
                if status in {"failed", "completed_with_error"}:
                    data["error"] = {"code": "server_error", "message": SECRET}
            self.response = httpx.Response(200, json=data, headers=response.headers, request=req)
            return self.response

    wire, events = StatusWire(), []

    async def authorize(_):
        return True

    async def record(event):
        events.append(event)
        return True

    engine = client(wire, authorize, record)
    try:
        with pytest.raises(GenerationEngineError) as error:
            await engine.generate_text(request())
        observation = error.value.observation
        assert error.value.failure.retryability.value == "no"
        assert observation.state.value != "completed"
        assert observation.provider_attempt_count == 1 and observation.retry_count == 0
        assert observation.provider_request_id == "req_synthetic"
        assert observation.provider_response_id == "resp_synthetic"
        assert observation.response_model == "writer-test"
        assert observation.input_tokens == 20 and observation.output_tokens == 8
        assert SECRET not in str(error.value)
        assert "Supported prose" not in observation.model_dump_json()
    finally:
        await engine.aclose()
    assert len(wire.calls) == 1 and wire.closed
    assert [e.outcome for e in events] == ["sending", "response_received"]
