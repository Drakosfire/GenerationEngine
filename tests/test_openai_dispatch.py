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
                                    "text": getattr(self, "text", "Supported prose [S1]."),
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
@pytest.mark.parametrize("stream_mode", [False, True])
async def test_callbacks_cannot_mutate_prepared_request(phase, mutation, stream_mode):
    wire, events = Wire(), []
    req = httpx.Request(
        "POST", ENDPOINT, json={"model": "writer-test", "tools": [], "stream": stream_mode}
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
        body["stream"] = "true"
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
async def test_guarded_stream_repair_and_structured_repair_refuse_without_io():
    wire = Wire()

    async def unexpected(_):
        pytest.fail("Unsupported capability reached gate")

    provider = OpenAITextProvider(
        api_key=SECRET, dispatch_gate=TextDispatchGate(unexpected, unexpected), transport=wire
    )
    try:
        events = [
            e
            async for e in provider.stream(TextGenerationCall(
                model="writer-test", user_prompt="x", json_schema={"type": "object"}
            ))
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


@pytest.mark.asyncio
@pytest.mark.parametrize("text,valid", [('{' + '"name":"Néra"' + '}', True), ('{"name":2}', False), ('not json', False)])
async def test_guarded_strict_schema_one_attempt_and_exact_capture(text, valid):
    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"], "additionalProperties": False}
    wire, views, events, attempts = Wire(), [], [], []
    wire.text = text
    async def authorize(view):
        views.append(view)
        return True
    async def record(event):
        events.append(event)
        return True
    async def completed(event):
        attempts.append(event)
    engine = client(wire, authorize, record)
    try:
        call = request(json_schema=schema, schema_name="neutral_record", max_conformance_retries=0)
        if valid:
            result = await engine.generate_structured(call, on_attempt_completed=completed)
            assert result.parsed == {"name": "Néra"}
            assert result.observation.provider_attempt_count == 1
            assert result.observation.conformance_retry_count == 0
        else:
            with pytest.raises(GenerationEngineError) as failure:
                await engine.generate_structured(call, on_attempt_completed=completed)
            assert failure.value.failure.code.value == "STRUCTURED_OUTPUT_INVALID"
            assert failure.value.observation.provider_attempt_count == 1
            assert failure.value.observation.input_tokens == 20
            assert failure.value.observation.conformance_retry_count == 0
    finally:
        await engine.aclose()
    assert len(wire.calls) == len(views) == len(attempts) == 1
    assert attempts[0].text == text
    assert views[0].body == wire.calls[0].content
    assert views[0].body_sha256 == hashlib.sha256(wire.calls[0].content).hexdigest()
    body = json.loads(views[0].body)
    assert body["text"]["format"] == {"type": "json_schema", "name": "neutral_record", "strict": True, "schema": schema}
    assert body["tools"] == [] and body["stream"] is False
    assert [event.outcome for event in events] == ["sending", "response_received"]
    assert wire.closed


@pytest.mark.asyncio
async def test_guarded_schema_refusal_is_not_repaired_or_misreported_as_empty():
    class RefusalWire(Wire):
        async def handle_async_request(self, req):
            response = await super().handle_async_request(req)
            body = json.loads(response.content)
            body["output"][0]["content"] = [{"type": "refusal", "refusal": "Synthetic refusal"}]
            return httpx.Response(200, json=body, request=req)
    async def acknowledge(_):
        return True
    wire = RefusalWire()
    engine = client(wire, acknowledge, acknowledge)
    try:
        with pytest.raises(GenerationEngineError) as error:
            await engine.generate_structured(request(json_schema={"type": "object"}, max_conformance_retries=0))
        assert error.value.failure.code.value == "PROVIDER_REFUSED"
        assert error.value.observation.input_tokens == 20
        assert error.value.observation.provider_attempt_count == 1
    finally:
        await engine.aclose()
    assert len(wire.calls) == 1


# Raw official-SDK SSE over a synthetic HTTP transport. No provider access.
SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}},
          "required": ["value"], "additionalProperties": False}


def stream_request(**changes):
    fields = dict(provider="openai", model="writer-test", user_prompt="synthetic",
                  temperature=None, max_transport_retries=0, max_conformance_retries=0,
                  json_schema=SCHEMA, schema_name="stream_value")
    fields.update(changes)
    return TextRequest(**fields)


def response_payload(text='{"value":"雪"}', **changes):
    result = dict(id="resp_stream", object="response", created_at=1, status="completed",
                  model="writer-test", output=[dict(type="message", id="msg_stream",
                  role="assistant", status="completed", content=[dict(type="output_text",
                  text=text, annotations=[])])],
                  usage=dict(input_tokens=20, input_tokens_details=dict(cached_tokens=0),
                             output_tokens=8, output_tokens_details=dict(reasoning_tokens=2),
                             total_tokens=28))
    result.update(changes)
    return result


def stream_events(text='{"value":"雪"}', terminal="response.completed", **response_changes):
    return [dict(type="response.created", sequence_number=0,
                 response=response_payload(output=[], status="in_progress", usage=None)),
            dict(type="response.output_text.delta", sequence_number=1, item_id="msg_stream",
                 output_index=0, content_index=0, delta=text[:5]),
            dict(type="response.output_text.delta", sequence_number=2, item_id="msg_stream",
                 output_index=0, content_index=0, delta=text[5:]),
            dict(type=terminal, sequence_number=3,
                 response=response_payload(text, **response_changes))]


class SSEBody(httpx.AsyncByteStream):
    def __init__(self, events, *, pause=False, broken=False):
        self.events = events
        self.pause = pause
        self.broken = broken
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, bytes):
                yield event
            else:
                yield ("data: " + json.dumps(event) + "\n\n").encode()
        if self.broken:
            raise httpx.ReadError(SECRET)
        if self.pause:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class SSEWire(httpx.AsyncBaseTransport):
    def __init__(self, body):
        self.body = body
        self.calls = []
        self.closed = False

    async def handle_async_request(self, req):
        self.calls.append(req)
        return httpx.Response(200, stream=self.body, request=req,
                              headers={"content-type": "text/event-stream",
                                       "x-request-id": "req_stream"})

    async def aclose(self):
        self.closed = True


async def capture_stream(wire, **changes):
    views, lifecycle = [], []

    async def authorize(view):
        views.append(view)
        return True

    async def record(event):
        lifecycle.append(event)
        return True

    engine = client(wire, authorize, record)
    try:
        events = [event async for event in engine.stream_text(stream_request(**changes))]
    finally:
        await engine.aclose()
    return events, views, lifecycle


@pytest.mark.asyncio
async def test_guarded_schema_stream_exact_sdk_bytes_and_real_deltas(monkeypatch):
    from generationengine import TextCompleted, TextDelta
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/ignored")
    body = SSEBody(stream_events())
    wire = SSEWire(body)
    events, views, lifecycle = await capture_stream(wire)
    assert len(wire.calls) == len(views) == 1
    sent = wire.calls[0]
    prepared = views[0]
    assert prepared.body == sent.content
    assert prepared.body_sha256 == hashlib.sha256(sent.content).hexdigest()
    assert prepared.body_size == len(sent.content)
    envelope = json.loads(sent.content)
    assert envelope["stream"] is True and envelope["tools"] == []
    assert envelope["text"]["format"] == dict(
        type="json_schema", name="stream_value", schema=SCHEMA, strict=True)
    assert str(sent.url) == ENDPOINT
    assert [e.outcome for e in lifecycle] == ["sending", "response_received"]
    assert lifecycle[-1].http_status == 200
    assert [e.text for e in events if isinstance(e, TextDelta)] == ['{"val', 'ue":"雪"}']
    assert isinstance(events[-1], TextCompleted)
    assert events[-1].final_text == '{"value":"雪"}'
    obs = events[-1].observation
    assert (obs.provider_request_id, obs.provider_response_id) == ("req_stream", "resp_stream")
    assert (obs.input_tokens, obs.cached_input_tokens, obs.output_tokens, obs.reasoning_tokens) == (20, 0, 8, 2)
    assert (obs.provider_attempt_count, obs.transport_retry_count, obs.conformance_retry_count) == (1, 0, 0)
    assert body.closed and wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal,changes,code", [
    ("response.failed", {"status": "failed"}, "PROVIDER_ERROR"),
    ("response.incomplete", {"status": "incomplete"}, "PROVIDER_INCOMPLETE"),
    ("response.cancelled", {"status": "cancelled"}, "PROVIDER_INCOMPLETE"),
    ("response.completed", {"status": "incomplete"}, "PROVIDER_INCOMPLETE"),
    ("response.completed", {"status": "unknown"}, "PROVIDER_INCOMPLETE"),
    ("response.completed", {"status": None}, "PROVIDER_INCOMPLETE"),
    ("response.completed", {"status": "failed", "error": {"code": "server_error", "message": SECRET}}, "PROVIDER_ERROR"),
    ("response.completed", {"output": []}, "PROVIDER_INCOMPLETE"),
])
async def test_guarded_stream_partial_valid_json_never_overrides_bad_terminal(terminal, changes, code):
    from generationengine import TextFailed
    wire = SSEWire(SSEBody(stream_events(terminal=terminal, **changes)))
    events, _, lifecycle = await capture_stream(wire)
    assert isinstance(events[-1], TextFailed)
    assert events[-1].failure.code.value == code
    assert events[-1].observation.provider_response_id == "resp_stream"
    assert events[-1].observation.input_tokens == 20
    assert SECRET not in events[-1].model_dump_json()
    assert len(wire.calls) == 1
    assert [e.outcome for e in lifecycle] == ["sending", "response_received"]
    assert wire.body.closed


@pytest.mark.asyncio
async def test_guarded_stream_terminal_refusal_preserves_usage_and_no_refusal_text():
    from generationengine import TextFailed
    refused = [dict(type="message", id="msg_stream", role="assistant", status="completed",
                    content=[dict(type="refusal", refusal=SECRET)])]
    wire = SSEWire(SSEBody(stream_events(output=refused)))
    events, _, _ = await capture_stream(wire)
    assert isinstance(events[-1], TextFailed)
    assert events[-1].failure.code.value == "PROVIDER_REFUSED"
    assert events[-1].observation.reasoning_tokens == 2
    assert SECRET not in events[-1].model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken,malformed", [(False, False), (True, False), (False, True)])
async def test_guarded_stream_body_failure_and_missing_terminal_keep_header_ack(broken, malformed):
    from generationengine import TextFailed
    raw = stream_events()[:-1]
    if malformed:
        raw.append(b'data: {not json\n\n')
    wire = SSEWire(SSEBody(raw, broken=broken))
    events, _, lifecycle = await capture_stream(wire)
    assert isinstance(events[-1], TextFailed)
    if not broken and not malformed:
        assert events[-1].failure.code.value == "STREAM_INCOMPLETE"
    assert events[-1].observation.provider_request_id == "req_stream"
    assert events[-1].observation.provider_response_id == "resp_stream"
    assert events[-1].observation.input_tokens is None
    assert [e.outcome for e in lifecycle] == ["sending", "response_received"]
    assert SECRET not in events[-1].model_dump_json()
    assert len(wire.calls) == 1 and wire.body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["deny", "sending", "response_received"])
async def test_guarded_stream_denied_callbacks_do_not_reissue(outcome):
    from generationengine import TextFailed
    wire = SSEWire(SSEBody(stream_events()))
    seen = []

    async def authorize(_):
        return outcome != "deny"

    async def record(e):
        seen.append(e.outcome)
        return e.outcome != outcome

    engine = client(wire, authorize, record)
    try:
        events = [e async for e in engine.stream_text(stream_request())]
    finally:
        await engine.aclose()
    assert isinstance(events[-1], TextFailed)
    assert events[-1].failure.code.value == "ATTEMPT_EVIDENCE_PERSISTENCE_FAILED"
    assert len(wire.calls) == (1 if outcome == "response_received" else 0)
    assert seen == (["sending", "response_received"] if outcome == "response_received" else
                    ["sending", "known_not_sent"] if outcome == "sending" else ["known_not_sent"])
    if wire.calls:
        assert wire.body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["early_close", "cancel", "timeout"])
async def test_guarded_stream_closes_live_body_on_retirement(finish):
    from generationengine import TextFailed
    body = SSEBody(stream_events()[:-1], pause=True)
    wire = SSEWire(body)
    seen = []

    async def authorize(_):
        return True

    async def record(e):
        seen.append(e.outcome)
        return True

    engine = client(wire, authorize, record)
    stream = engine.stream_text(stream_request(deadline_ms=50 if finish == "timeout" else None))
    try:
        await anext(stream)
        if finish == "early_close":
            await stream.aclose()
        elif finish == "cancel":
            await anext(stream)
            task = asyncio.create_task(anext(stream))
            await asyncio.wait_for(body.waiting.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            rest = [e async for e in stream]
            assert isinstance(rest[-1], TextFailed)
            assert rest[-1].failure.code.value == "PROVIDER_TIMEOUT"
        assert body.closed
        assert seen == ["sending", "response_received"]
        assert len(wire.calls) == 1
    finally:
        await stream.aclose()
        await engine.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ['{"value":2}', 'not JSON'])
async def test_real_guarded_stream_final_schema_failure_does_not_repair(text):
    from generationengine import TextFailed
    wire = SSEWire(SSEBody(stream_events(text)))
    events, _, lifecycle = await capture_stream(wire)
    assert isinstance(events[-1], TextFailed)
    assert events[-1].failure.code.value == "STRUCTURED_OUTPUT_INVALID"
    assert events[-1].observation.input_tokens == 20
    assert events[-1].observation.provider_response_id == "resp_stream"
    assert len(wire.calls) == 1 and len(lifecycle) == 2
    assert wire.body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel"])
async def test_guarded_stream_transport_failure_before_headers_is_uncertain(outcome):
    class BrokenWire(SSEWire):
        async def handle_async_request(self, req):
            self.calls.append(req)
            if outcome == "cancel":
                raise asyncio.CancelledError()
            raise httpx.ReadTimeout(SECRET, request=req)

    wire = BrokenWire(SSEBody([]))
    seen = []

    async def authorize(_):
        return True

    async def record(e):
        seen.append(e.outcome)
        return True

    engine = client(wire, authorize, record)
    try:
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                _ = [e async for e in engine.stream_text(stream_request())]
        else:
            events = [e async for e in engine.stream_text(stream_request())]
            assert events[-1].failure.code.value == "PROVIDER_TIMEOUT"
            assert SECRET not in events[-1].model_dump_json()
        assert len(wire.calls) == 1
        assert seen == ["sending", "outcome_unknown"]
    finally:
        await engine.aclose()


class PhaseWire(Wire):
    def __init__(self, output):
        super().__init__()
        self.output = output

    async def handle_async_request(self, req):
        response = await super().handle_async_request(req)
        body = json.loads(response.content)
        body["output"] = self.output
        self.response = httpx.Response(200, json=body, headers=response.headers, request=req)
        return self.response


def phase_message(phase, text, *, message_id="msg_answer"):
    return dict(type="message", id=message_id, role="assistant", status="completed",
                phase=phase, content=[dict(type="output_text", text=text, annotations=[])])


@pytest.mark.asyncio
@pytest.mark.parametrize("guarded", [False, True])
async def test_final_phase_excludes_commentary_before_structured_conformance(guarded):
    from openai import AsyncOpenAI

    wire = PhaseWire([
        phase_message("commentary", '{"value":"draft"}<|fim_suffix|>', message_id="msg_comment"),
        phase_message("final_answer", '{"value":"supported"}'),
    ])
    views, lifecycle, attempts = [], [], []
    async def authorize(view):
        views.append(view)
        return True
    async def record(event):
        lifecycle.append(event)
        return True
    async def completed(event):
        attempts.append(event)
    if guarded:
        engine = client(wire, authorize, record)
    else:
        sdk = AsyncOpenAI(api_key=SECRET, max_retries=0,
                          http_client=httpx.AsyncClient(transport=wire))
        engine = GenerationClient(text_providers={"openai": OpenAITextProvider(client=sdk)})
    try:
        result = await engine.generate_structured(
            request(json_schema=SCHEMA, max_conformance_retries=0), on_attempt_completed=completed)
    finally:
        await engine.aclose()
    assert result.parsed == {"value": "supported"}
    assert attempts[0].text == '{"value":"supported"}'
    assert len(wire.calls) == len(attempts) == 1
    assert (result.observation.provider_request_id, result.observation.provider_response_id) == (
        "req_synthetic", "resp_synthetic")
    assert (result.observation.input_tokens, result.observation.output_tokens) == (20, 8)
    assert result.observation.conformance_retry_count == 0
    if guarded:
        assert len(views) == 1 and views[0].body == wire.calls[0].content
        assert [event.outcome for event in lifecycle] == ["sending", "response_received"]
    assert wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("output,code", [
    ([phase_message(None, "one"), phase_message(None, "two")], "MALFORMED_PROVIDER_RESPONSE"),
    ([phase_message("commentary", "not a final")], "PROVIDER_INCOMPLETE"),
    ([], "PROVIDER_INCOMPLETE"),
    ([phase_message("final_answer", "one"), phase_message("final_answer", "two")], "MALFORMED_PROVIDER_RESPONSE"),
    ([phase_message("unknown", "one")], "MALFORMED_PROVIDER_RESPONSE"),
    ([phase_message(None, "one"), phase_message("final_answer", "two")], "MALFORMED_PROVIDER_RESPONSE"),
    ([phase_message("final_answer", "  ")], "PROVIDER_INCOMPLETE"),
    ([dict(phase_message("final_answer", "partial"), status="incomplete")], "PROVIDER_INCOMPLETE"),
])
async def test_ambiguous_or_missing_final_retains_accounting_without_retry(output, code):
    wire, lifecycle = PhaseWire(output), []
    async def authorize(_):
        return True
    async def record(event):
        lifecycle.append(event)
        return True
    engine = client(wire, authorize, record)
    try:
        with pytest.raises(GenerationEngineError) as error:
            await engine.generate_text(request())
    finally:
        await engine.aclose()
    assert error.value.failure.code.value == code
    assert error.value.failure.retryability.value == "no"
    obs = error.value.observation
    assert obs.state.value != "completed"
    assert (obs.provider_request_id, obs.provider_response_id) == ("req_synthetic", "resp_synthetic")
    assert (obs.input_tokens, obs.output_tokens) == (20, 8)
    assert obs.provider_attempt_count == len(wire.calls) == 1
    assert [event.outcome for event in lifecycle] == ["sending", "response_received"]
    assert "not a final" not in error.value.failure.message


@pytest.mark.asyncio
async def test_refusal_in_commentary_precedes_final_text():
    comment = phase_message("commentary", "unused")
    comment["content"] = [dict(type="refusal", refusal="private refusal")]
    wire = PhaseWire([comment, phase_message("final_answer", "must not deliver")])
    async def authorize(_):
        return True
    async def record(_):
        return True
    engine = client(wire, authorize, record)
    try:
        with pytest.raises(GenerationEngineError) as error:
            await engine.generate_text(request())
    finally:
        await engine.aclose()
    assert error.value.failure.code.value == "PROVIDER_REFUSED"
    assert error.value.observation.input_tokens == 20
    assert "private refusal" not in error.value.failure.message
    assert len(wire.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [None, "final_answer"])
async def test_single_answer_joins_its_own_text_parts_in_order(phase):
    message = phase_message(phase, "first ")
    message["content"].append(dict(type="output_text", text="second", annotations=[]))
    wire = PhaseWire([message])
    async def authorize(_):
        return True
    async def record(_):
        return True
    engine = client(wire, authorize, record)
    try:
        result = await engine.generate_text(request())
    finally:
        await engine.aclose()
    assert result.text == "first second"
    assert len(wire.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_stream_terminal_selects_final_and_never_promotes_provisional_text(invalid):
    from generationengine import TextCompleted, TextFailed

    output = [phase_message("commentary", 'draft<|fim_suffix|>')]
    if not invalid:
        output.append(phase_message("final_answer", '{"value":"supported"}'))
    body = SSEBody(stream_events(output=output))
    wire = SSEWire(body)
    events, views, lifecycle = await capture_stream(wire)
    assert len(wire.calls) == len(views) == 1
    assert [event.outcome for event in lifecycle] == ["sending", "response_received"]
    terminal = events[-1]
    if invalid:
        assert isinstance(terminal, TextFailed)
        assert terminal.failure.code.value == "PROVIDER_INCOMPLETE"
        assert not any(isinstance(event, TextCompleted) for event in events)
    else:
        assert isinstance(terminal, TextCompleted)
        assert terminal.final_text == '{"value":"supported"}'
    assert terminal.observation.input_tokens == 20
    assert terminal.observation.provider_response_id == "resp_stream"
    assert body.closed and wire.closed
