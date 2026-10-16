"""Opt-in authorization for an immutable prepared text request."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PreparedTextDispatch:
    attempt_id: str
    provider: str
    api_mode: str
    model: str
    method: str
    url: str
    body: bytes
    body_sha256: str
    body_size: int


@dataclass(frozen=True)
class TextDispatchEvent:
    request: PreparedTextDispatch
    outcome: Literal["sending", "known_not_sent", "response_received", "outcome_unknown"]
    http_status: int | None = None


@dataclass(frozen=True)
class TextDispatchGate:
    """Consumer callbacks must explicitly acknowledge with True.

    Body bytes may contain caller data. Headers/credentials are never exposed.
    The consumer owns authorization, persistence and uncertain-outcome recovery.
    """

    authorize: Callable[[PreparedTextDispatch], Awaitable[bool]]
    record: Callable[[TextDispatchEvent], Awaitable[bool]]
