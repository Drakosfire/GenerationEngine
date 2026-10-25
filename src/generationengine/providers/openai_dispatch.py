"""OpenAI Responses wire gate using supported SDK http_client injection."""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from uuid import uuid4

import httpx

from generationengine.failures import FailureCode
from generationengine.providers.errors import ProviderError
from generationengine.text_dispatch import (
    PreparedTextDispatch,
    TextDispatchEvent,
    TextDispatchGate,
)

_ENDPOINT = "https://api.openai.com/v1/responses"


def _gate_error() -> ProviderError:
    return ProviderError.from_code(
        FailureCode.ATTEMPT_EVIDENCE_PERSISTENCE_FAILED,
        "Text dispatch authorization or lifecycle was not acknowledged.",
    )


class GuardedOpenAITransport(httpx.AsyncBaseTransport):
    def __init__(self, gate: TextDispatchGate, transport: httpx.AsyncBaseTransport) -> None:
        self.gate = gate
        self.transport = transport

    async def _record(
        self,
        prepared: PreparedTextDispatch,
        outcome: Literal["sending", "known_not_sent", "response_received", "outcome_unknown"],
        status: int | None = None,
    ) -> None:
        if await self.gate.record(TextDispatchEvent(prepared, outcome, status)) is not True:
            raise _gate_error()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
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
                or type(payload.get("stream")) is not bool
                or payload.get("tools") != []
                or not isinstance(model, str)
                or not model
            ):
                raise ValueError("Unexpected prepared text identity")
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError.from_code(
                FailureCode.INVALID_REQUEST,
                "Unsupported prepared text dispatch.",
            ) from exc
        prepared = PreparedTextDispatch(
            str(uuid4()),
            "openai",
            "responses",
            model,
            request.method,
            str(request.url),
            body,
            hashlib.sha256(body).hexdigest(),
            len(body),
        )
        original_headers = tuple(request.headers.multi_items())
        original_stream = request.stream
        wire_request = httpx.Request(
            prepared.method,
            prepared.url,
            headers=original_headers,
            content=prepared.body,
            extensions=dict(request.extensions),
        )
        try:
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
