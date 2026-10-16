"""Actual SDK serialization → opt-in gate → mock wire, never live I/O."""

import asyncio
import hashlib
import json
from dataclasses import FrozenInstanceError

import httpx2
import pytest
from test_typesafe_decision import request, sdk_response

from generationengine import GenerationClient, GenerationEngineError
from generationengine.decision_dispatch import DecisionDispatchGate
from generationengine.providers.typesafe_decision import TypeSafeDecisionProvider
from generationengine.providers.typesafe_dispatch import GuardedTypeSafeTransport


class Wire(httpx2.AsyncBaseTransport):
    def __init__(self, outcome=None):
        self.calls = []
        self.closed = False
        self.outcome = outcome

    async def handle_async_request(self, req):
        self.calls.append(req)
        if self.outcome == "timeout":
            raise httpx2.ReadTimeout("private detail", request=req)
        if self.outcome == "cancel":
            raise asyncio.CancelledError()
        payload = (
            {"bad": True}
            if self.outcome == "malformed"
            else sdk_response(model="jev-1.14.0").model_dump(mode="json")
        )
        return httpx2.Response(200, json=payload, request=req)

    async def aclose(self):
        self.closed = True


def provider(wire, authorize, record):
    return TypeSafeDecisionProvider(
        api_key="credential-not-for-observer",
        dispatch_gate=DecisionDispatchGate(authorize, record),
        transport=wire,
    )


@pytest.mark.asyncio
async def test_actual_sdk_wire_bytes_unicode_question_types_and_secret_exclusion():
    wire, prepared, events = Wire(), [], []

    async def authorize(view):
        prepared.append(view)
        with pytest.raises(FrozenInstanceError):
            view.model = "changed"
        return True

    async def record(event):
        events.append(event)
        return True

    client = GenerationClient(decision_providers={"typesafe": provider(wire, authorize, record)})
    req = request().model_copy(
        update={"model": "jev-1.14.0", "state": {"quote": 'Néra says "yes"\n雪'}}
    )
    try:
        result = await client.decide(req)
    finally:
        await client.aclose()
    assert result.observation.response_model == "jev-1.14.0"
    assert len(prepared) == len(wire.calls) == 1
    view = prepared[0]
    sent = wire.calls[0]
    assert view.body == sent.content
    assert view.body_sha256 == hashlib.sha256(sent.content).hexdigest()
    assert view.body_size == len(sent.content)
    assert json.loads(sent.content)["model"] == "jev-1.14.0"
    assert "Néra" in sent.content.decode() or "N\\u00e9ra" in sent.content.decode()
    assert "content-encoding" not in sent.headers
    assert sent.headers["authorization"] == "Bearer credential-not-for-observer"
    assert "credential-not-for-observer" not in repr(view)
    assert "headers" not in vars(view)
    assert [e.outcome for e in events] == ["sending", "response_received"]
    assert events[-1].http_status == 200
    assert wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["deny", "raise", "cancel", "record_fail"])
async def test_rejection_before_send_never_reaches_network(action):
    wire, events = Wire(), []

    async def authorize(view):
        if action == "raise":
            raise RuntimeError("credential-not-for-observer")
        if action == "cancel":
            raise asyncio.CancelledError()
        return action != "deny"

    async def record(event):
        events.append(event)
        return not (action == "record_fail" and event.outcome == "sending")

    client = GenerationClient(decision_providers={"typesafe": provider(wire, authorize, record)})
    try:
        with pytest.raises(asyncio.CancelledError if action == "cancel" else GenerationEngineError):
            await client.decide(
                request().model_copy(update={"model": "jev-1.14.0", "max_transport_retries": 2})
            )
    finally:
        await client.aclose()
    assert wire.calls == []
    assert events[-1].outcome == "known_not_sent"
    assert wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel", "malformed"])
async def test_real_send_failure_is_not_retried_and_response_ack_survives_parse_error(outcome):
    wire, events = Wire(outcome), []

    async def authorize(view):
        return True

    async def record(event):
        events.append(event)
        return True

    client = GenerationClient(decision_providers={"typesafe": provider(wire, authorize, record)})
    try:
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancel" else GenerationEngineError
        ) as failure:
            await client.decide(
                request().model_copy(update={"model": "jev-1.14.0", "max_transport_retries": 2})
            )
        if outcome != "cancel":
            assert failure.value.observation.provider_attempt_count == 1
    finally:
        await client.aclose()
    assert len(wire.calls) == 1
    assert [e.outcome for e in events] == [
        "sending",
        "response_received" if outcome == "malformed" else "outcome_unknown",
    ]
    assert wire.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["url", "method", "body", "headers"])
async def test_mutation_after_authorization_is_denied(mutation):
    wire, events = Wire(), []
    req = httpx2.Request(
        "POST", "https://ai-gateway.vercel.sh/typesafe/v1/systemone", json={"model": "jev-1.14.0"}
    )

    async def authorize(view):
        if mutation == "url":
            req.url = httpx2.URL("https://example.invalid/steal")
        elif mutation == "method":
            req.method = "GET"
        elif mutation == "body":
            req.stream = httpx2.ByteStream(b"changed")
        else:
            req.headers["authorization"] = "Bearer changed"
        return True

    async def record(event):
        events.append(event)
        return True

    transport = GuardedTypeSafeTransport(DecisionDispatchGate(authorize, record), wire)
    with pytest.raises(Exception):
        await transport.handle_async_request(req)
    assert not wire.calls and events[-1].outcome == "known_not_sent"
    await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, [], {}, {"model": ""}, {"model": 7}])
async def test_invalid_wire_identity_never_requests_authorization(payload):
    wire = Wire()

    async def unexpected(_):
        pytest.fail("Invalid requests must not reach authorization or lifecycle")

    transport = GuardedTypeSafeTransport(DecisionDispatchGate(unexpected, unexpected), wire)
    req = httpx2.Request("POST", "https://ai-gateway.vercel.sh/typesafe/v1/systemone", json=payload)
    from generationengine.providers.errors import ProviderError

    with pytest.raises(ProviderError):
        await transport.handle_async_request(req)
    assert wire.calls == []
    await transport.aclose()


@pytest.mark.asyncio
async def test_lifecycle_callback_cannot_mutate_request_after_authorization():
    wire, events = Wire(), []
    req = httpx2.Request(
        "POST", "https://ai-gateway.vercel.sh/typesafe/v1/systemone", json={"model": "jev-1.14.0"}
    )

    async def authorize(_):
        return True

    async def record(event):
        events.append(event)
        if event.outcome == "sending":
            req.headers["authorization"] = "changed"
        return True

    transport = GuardedTypeSafeTransport(DecisionDispatchGate(authorize, record), wire)
    from generationengine.providers.errors import ProviderError

    with pytest.raises(ProviderError):
        await transport.handle_async_request(req)
    assert wire.calls == []
    assert [e.outcome for e in events] == ["sending", "known_not_sent"]
    await transport.aclose()


@pytest.mark.asyncio
async def test_unacknowledged_response_record_closes_response_without_retry():
    class ResponseWire(Wire):
        async def handle_async_request(self, req):
            self.response = await super().handle_async_request(req)
            return self.response

    wire, events = ResponseWire(), []

    async def authorize(_):
        return True

    async def record(event):
        events.append(event)
        return event.outcome != "response_received"

    client = GenerationClient(decision_providers={"typesafe": provider(wire, authorize, record)})
    try:
        with pytest.raises(GenerationEngineError):
            await client.decide(
                request().model_copy(update={"model": "jev-1.14.0", "max_transport_retries": 2})
            )
        assert len(wire.calls) == 1
        assert wire.response.is_closed
        assert [e.outcome for e in events] == ["sending", "response_received"]
    finally:
        await client.aclose()
    assert wire.closed


@pytest.mark.asyncio
async def test_concurrent_sdk_calls_keep_distinct_authorized_bytes_and_lifecycle():
    wire, prepared, events = Wire(), [], []
    ready = asyncio.Event()

    async def authorize(view):
        prepared.append(view)
        if len(prepared) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=2)
        return True

    async def record(event):
        events.append(event)
        return True

    client = GenerationClient(decision_providers={"typesafe": provider(wire, authorize, record)})
    try:
        await asyncio.gather(
            *[
                client.decide(
                    request().model_copy(update={"model": "jev-1.14.0", "state": {"case": name}})
                )
                for name in ["first", "second"]
            ]
        )
    finally:
        await client.aclose()
    assert len(wire.calls) == len(prepared) == 2
    assert len({p.attempt_id for p in prepared}) == 2
    assert {p.body for p in prepared} == {r.content for r in wire.calls}
    assert {json.loads(p.body)["state"]["case"] for p in prepared} == {"first", "second"}
    for view in prepared:
        assert [e.outcome for e in events if e.request.attempt_id == view.attempt_id] == [
            "sending",
            "response_received",
        ]
    assert wire.closed
