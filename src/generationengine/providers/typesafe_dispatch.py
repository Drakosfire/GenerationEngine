"""TypeSafe wire gate using the SDK's supported httpx2 transport injection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Literal
from uuid import uuid4

import httpx2

from generationengine.decision_dispatch import (
    DecisionContextMode,
    DecisionDispatchEvent,
    DecisionDispatchGate,
    PreparedDecisionDispatch,
)
from generationengine.failures import FailureCode
from generationengine.providers.errors import ProviderError

_LONG_CONTEXT_HEADER = "TypeSafe-Danger-Zone-Long-Context"

_ENDPOINT = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"


def _gate_error() -> ProviderError:
    return ProviderError.from_code(
        FailureCode.ATTEMPT_EVIDENCE_PERSISTENCE_FAILED,
        "Decision dispatch authorization or lifecycle was not acknowledged.",
        provider_transport="vercel_ai_gateway",
    )


class GuardedTypeSafeTransport(httpx2.AsyncBaseTransport):
    def __init__(self, gate: DecisionDispatchGate, transport: httpx2.AsyncBaseTransport) -> None:
        self.gate = gate
        self.transport = transport

    async def _record(
        self,
        prepared: PreparedDecisionDispatch,
        outcome: Literal["sending", "known_not_sent", "response_received", "outcome_unknown"],
        status: int | None = None,
    ) -> None:
        if await self.gate.record(DecisionDispatchEvent(prepared, outcome, status)) is not True:
            raise _gate_error()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = await request.aread()
        # No redirects, alternate endpoint, compressed/streamed replacement body,
        # user-info or query-string destination can inherit this authorization.
        try:
            payload = json.loads(body)
            model = payload["model"]
            if (
                request.method != "POST"
                or str(request.url) != _ENDPOINT
                or request.headers.get("content-encoding")
                or _LONG_CONTEXT_HEADER in request.headers
                or not isinstance(model, str)
                or not model
            ):
                raise ValueError("Unexpected prepared decision identity")
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError.from_code(
                FailureCode.INVALID_REQUEST,
                "Unsupported prepared decision dispatch.",
                provider_transport="vercel_ai_gateway",
            ) from exc
        prepared = PreparedDecisionDispatch(
            str(uuid4()),
            "typesafe",
            "vercel_ai_gateway",
            model,
            request.method,
            str(request.url),
            body,
            hashlib.sha256(body).hexdigest(),
            len(body),
        )
        original_headers = tuple(request.headers.multi_items())
        original_stream = request.stream
        try:
            if self.gate.select_context is not None:
                mode = await self.gate.select_context(prepared)
                if type(mode) is not DecisionContextMode:
                    raise _gate_error()
                prepared = replace(prepared, context_mode=mode)
            wire_headers = list(original_headers)
            if prepared.context_mode is DecisionContextMode.LONG_CONTEXT:
                wire_headers.append((_LONG_CONTEXT_HEADER, "true"))
            wire_request = httpx2.Request(
                prepared.method,
                prepared.url,
                headers=wire_headers,
                content=prepared.body,
                extensions=dict(request.extensions),
            )
            if await self.gate.authorize(prepared) is not True:
                raise _gate_error()
            if (
                request.method != prepared.method
                or str(request.url) != prepared.url
                or request.content != prepared.body
                or request.stream is not original_stream
                or tuple(request.headers.multi_items()) != original_headers
            ):
                raise _gate_error()
            await self._record(prepared, "sending")
            # The lifecycle callback also cannot change approved request bytes.
            if (
                request.method != prepared.method
                or str(request.url) != prepared.url
                or request.content != prepared.body
                or request.stream is not original_stream
                or tuple(request.headers.multi_items()) != original_headers
            ):
                raise _gate_error()
        except BaseException:
            await self._record(prepared, "known_not_sent")
            raise
        try:
            response = await self.transport.handle_async_request(wire_request)
        except BaseException:
            await self._record(prepared, "outcome_unknown")
            raise
        try:
            await self._record(prepared, "response_received", response.status_code)
        except BaseException:
            await response.aclose()
            raise
        return response

    async def aclose(self) -> None:
        await self.transport.aclose()
