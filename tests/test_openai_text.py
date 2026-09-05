"""OpenAI adapter: SDK retry ownership and request vs response IDs."""

from __future__ import annotations

from types import SimpleNamespace

from generationengine.providers.openai_text import OpenAITextProvider, _ids_from_response


def test_openai_client_disables_sdk_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "generationengine.providers.openai_text.AsyncOpenAI",
        FakeAsyncOpenAI,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    OpenAITextProvider()
    assert captured["api_key"] == "sk-test"
    assert captured["max_retries"] == 0


def test_request_id_is_http_id_not_response_object_id() -> None:
    response = SimpleNamespace(
        id="resp_abc",
        _request_id="req_http",
        output_text="ok",
        model="gpt-5.1",
        usage=None,
        refusal=None,
    )
    request_id, response_id = _ids_from_response(response)
    assert request_id == "req_http"
    assert response_id == "resp_abc"
    result = OpenAITextProvider(client=SimpleNamespace())._result_from_response(
        response, structured=False
    )
    assert result.provider_request_id == "req_http"
    assert result.provider_response_id == "resp_abc"


def test_openai_maps_sdk_errors_to_safe_public_messages() -> None:
    from generationengine.failures import FailureCode

    provider = OpenAITextProvider(client=SimpleNamespace())
    error = provider._map_exception(
        RuntimeError("Authorization Bearer sk-live HTTP 502 from api.openai.com")
    )
    assert error.failure.code is FailureCode.PROVIDER_ERROR
    assert error.failure.message == "Provider request failed."
    assert "sk-live" not in error.failure.message
    assert "openai.com" not in error.failure.message

    class FakeTimeoutError(Exception):
        pass

    timeout = provider._map_exception(FakeTimeoutError("waited 45s"))
    assert timeout.failure.code is FailureCode.PROVIDER_TIMEOUT
    assert timeout.failure.message == "Provider request timed out."

    class FakeRateLimitError(Exception):
        pass

    rate_limited = provider._map_exception(
        FakeRateLimitError(
            "429 https://api.openai.com/v1/responses Authorization Bearer sk-live"
        )
    )
    assert rate_limited.failure.code is FailureCode.RATE_LIMITED
    assert rate_limited.failure.message == "Provider rate limit exceeded."
    assert "sk-live" not in rate_limited.failure.message
    assert "openai.com" not in rate_limited.failure.message
