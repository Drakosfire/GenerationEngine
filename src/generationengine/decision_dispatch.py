"""Opt-in product authorization over a provider's immutable prepared request."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PreparedDecisionDispatch:
    attempt_id: str
    provider: str
    provider_transport: str
    model: str
    method: str
    url: str
    body: bytes
    body_sha256: str
    body_size: int


@dataclass(frozen=True)
class DecisionDispatchEvent:
    request: PreparedDecisionDispatch
    outcome: Literal["sending", "known_not_sent", "response_received", "outcome_unknown"]
    http_status: int | None = None


@dataclass(frozen=True)
class DecisionDispatchGate:
    """Callbacks are product-owned; only explicit True acknowledges either gate.

    Body bytes contain caller data. Secret headers are never provided. Callbacks
    must persist their own authority/lifecycle and must not mutate shared state.
    """

    authorize: Callable[[PreparedDecisionDispatch], Awaitable[bool]]
    record: Callable[[DecisionDispatchEvent], Awaitable[bool]]
