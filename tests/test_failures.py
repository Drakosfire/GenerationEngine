"""Normalized failure taxonomy tests."""

from __future__ import annotations

from generationengine.failures import (
    FAILURE_RETRYABILITY,
    FailureCode,
    InferenceFailure,
    Retryability,
)


def test_all_accepted_failure_codes_exist() -> None:
    expected = {
        "CONFIGURATION_UNAVAILABLE",
        "UNSUPPORTED_CAPABILITY",
        "INVALID_REQUEST",
        "PROVIDER_REFUSED",
        "RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_UNAVAILABLE",
        "PROVIDER_ERROR",
        "MALFORMED_PROVIDER_RESPONSE",
        "STRUCTURED_OUTPUT_INVALID",
        "STREAM_INCOMPLETE",
        "INTERNAL_ERROR",
    }
    assert {code.value for code in FailureCode} == expected
    assert set(FAILURE_RETRYABILITY) == set(FailureCode)


def test_retryability_supports_yes_no_unknown() -> None:
    assert set(Retryability) == {Retryability.YES, Retryability.NO, Retryability.UNKNOWN}
    assert FAILURE_RETRYABILITY[FailureCode.RATE_LIMITED] is Retryability.YES
    assert FAILURE_RETRYABILITY[FailureCode.INVALID_REQUEST] is Retryability.NO
    assert FAILURE_RETRYABILITY[FailureCode.PROVIDER_ERROR] is Retryability.UNKNOWN


def test_public_failure_does_not_require_provider_sdk_exceptions() -> None:
    failure = InferenceFailure.from_code(
        FailureCode.PROVIDER_UNAVAILABLE,
        "provider overloaded",
    )
    assert failure.code is FailureCode.PROVIDER_UNAVAILABLE
    assert failure.retryability is Retryability.YES
    assert "openai" not in failure.model_dump_json().lower()
    dumped = failure.model_dump()
    assert set(dumped) == {"code", "message", "retryability"}
