"""InferenceObservation contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from generationengine.failures import FailureCode
from generationengine.observation import (
    FORBIDDEN_OBSERVATION_FIELDS,
    InferenceObservation,
    ObservationState,
)


def test_unknown_usage_is_none_not_zero() -> None:
    observation = InferenceObservation(
        latency_ms=10,
        retry_count=0,
        state=ObservationState.COMPLETED,
    )
    assert observation.input_tokens is None
    assert observation.cached_input_tokens is None
    assert observation.output_tokens is None
    assert observation.cost_usd is None
    assert observation.provider_request_id is None


def test_explicit_zero_usage_is_zero() -> None:
    observation = InferenceObservation(
        input_tokens=0,
        cached_input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        latency_ms=0,
        retry_count=0,
        state=ObservationState.COMPLETED,
    )
    assert observation.input_tokens == 0
    assert observation.output_tokens == 0
    assert observation.cost_usd == 0.0


def test_retry_count_is_additional_attempts() -> None:
    """Three total attempts means two retries."""
    observation = InferenceObservation(
        latency_ms=100,
        retry_count=2,
        state=ObservationState.COMPLETED,
    )
    assert observation.retry_count == 2


def test_completed_cannot_include_failure_code() -> None:
    with pytest.raises(ValidationError):
        InferenceObservation(
            latency_ms=1,
            retry_count=0,
            state=ObservationState.COMPLETED,
            failure_code=FailureCode.INTERNAL_ERROR,
        )


def test_failed_requires_failure_code() -> None:
    with pytest.raises(ValidationError):
        InferenceObservation(
            latency_ms=1,
            retry_count=0,
            state=ObservationState.FAILED,
        )
    observation = InferenceObservation(
        latency_ms=1,
        retry_count=1,
        state=ObservationState.FAILED,
        failure_code=FailureCode.PROVIDER_TIMEOUT,
    )
    assert observation.failure_code is FailureCode.PROVIDER_TIMEOUT


def test_non_negative_latency_and_retries() -> None:
    with pytest.raises(ValidationError):
        InferenceObservation(latency_ms=-1, retry_count=0, state=ObservationState.COMPLETED)
    with pytest.raises(ValidationError):
        InferenceObservation(latency_ms=0, retry_count=-1, state=ObservationState.COMPLETED)


def test_observation_model_has_no_prompt_payload_fields() -> None:
    field_names = set(InferenceObservation.model_fields)
    overlap = field_names & FORBIDDEN_OBSERVATION_FIELDS
    assert overlap == set()
