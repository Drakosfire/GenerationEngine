"""Provider-independent structured conformance for generate_structured()."""

from __future__ import annotations

import pytest

from generationengine import (
    FailureCode,
    GenerationClient,
    GenerationEngineError,
    InferenceProfile,
    ObservationState,
    TextGenerationCall,
    TextGenerationResult,
    TextRequest,
)
from generationengine.providers.errors import ProviderError

FIXTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["name", "count"],
    "additionalProperties": False,
}


class ScriptedText:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[TextGenerationCall] = []

    async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
        self.calls.append(call)
        if not self.outcomes:
            raise AssertionError("unexpected extra provider call")
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, call: TextGenerationCall):
        raise NotImplementedError


def _result(**kwargs) -> TextGenerationResult:
    defaults = {
        "text": '{"name":"ok","count":1}',
        "parsed": {"name": "ok", "count": 1},
        "provider_request_id": "req",
        "provider_response_id": "resp",
        "response_model": "model",
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 4,
    }
    defaults.update(kwargs)
    return TextGenerationResult(**defaults)


def _request(**kwargs) -> TextRequest:
    defaults = {
        "user_prompt": "make fixture",
        "profile": InferenceProfile.STRUCTURED_LOW_COST,
        "json_schema": FIXTURE_SCHEMA,
        "schema_name": "fixture",
    }
    defaults.update(kwargs)
    return TextRequest(**defaults)


@pytest.mark.asyncio
async def test_native_parsed_candidate_is_still_locally_validated() -> None:
    provider = ScriptedText(
        [
            _result(parsed={"name": 1, "count": "nope"}, text='{"name":1,"count":"nope"}'),
            _result(),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(_request())
    assert result.parsed == {"name": "ok", "count": 1}
    assert provider.calls[1].user_prompt.endswith(
        "Return corrected JSON that satisfies the supplied schema."
    ) or "did not satisfy the required schema" in provider.calls[1].user_prompt
    assert result.observation.conformance_retry_count == 1
    assert result.observation.retry_count == 0
    assert result.observation.transport_retry_count == 0
    assert result.observation.provider_attempt_count == 2


@pytest.mark.asyncio
async def test_non_native_invalid_then_valid_json() -> None:
    provider = ScriptedText(
        [
            _result(text="not-json", parsed=None, input_tokens=3, output_tokens=1),
            _result(text='{"name":"x","count":2}', parsed=None, input_tokens=5, output_tokens=2),
        ]
    )
    client = GenerationClient(
        text_providers={"openrouter": provider},
    )
    result = await client.generate_structured(
        _request(profile=None, provider="openrouter", model="deepseek/deepseek-v4.1-flash")
    )
    assert result.parsed == {"name": "x", "count": 2}
    assert result.observation.provider == "openrouter"
    assert result.observation.conformance_retry_count == 1
    assert result.observation.retry_count == 0
    assert result.observation.transport_retry_count == 0
    assert result.observation.provider_attempt_count == 2
    assert result.observation.input_tokens == 8
    assert result.observation.output_tokens == 3


@pytest.mark.asyncio
async def test_persistent_invalid_output_is_structured_invalid() -> None:
    provider = ScriptedText(
        [
            _result(
                text="{",
                parsed=None,
                provider_request_id="req-1",
                provider_response_id="resp-1",
                input_tokens=3,
                output_tokens=1,
            ),
            _result(
                text='{"name":1}',
                parsed=None,
                provider_request_id="req-2",
                provider_response_id="resp-2",
                input_tokens=4,
                output_tokens=2,
            ),
        ]
    )
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_structured(_request())
    assert exc.value.failure.code is FailureCode.STRUCTURED_OUTPUT_INVALID
    assert exc.value.observation.state is ObservationState.INCOMPLETE
    assert exc.value.observation.conformance_retry_count == 1
    assert exc.value.observation.provider_attempt_count == 2
    assert exc.value.observation.provider_request_id == "req-2"
    assert exc.value.observation.provider_response_id == "resp-2"
    assert exc.value.observation.response_model == "model"
    assert exc.value.observation.input_tokens == 7
    assert exc.value.observation.cached_input_tokens == 0
    assert exc.value.observation.output_tokens == 3
    dumped = exc.value.observation.model_dump()
    assert "user_prompt" not in dumped
    assert "prompt" not in dumped
    assert "{" not in str(dumped.get("failure_code"))


@pytest.mark.asyncio
async def test_incomplete_provider_error_stops_before_structured_repair() -> None:
    provider = ScriptedText([
        ProviderError.from_code(
            FailureCode.PROVIDER_INCOMPLETE,
            "partial raw output",
            provider_request_id="req-incomplete",
            provider_response_id="resp-incomplete",
            response_model="model",
            input_tokens=7,
            cached_input_tokens=0,
            output_tokens=3,
            reasoning_tokens=2,
        )
    ])
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient(text_provider=provider).generate_structured(
            _request(max_transport_retries=3)
        )
    assert len(provider.calls) == 1
    assert exc.value.failure.code is FailureCode.PROVIDER_INCOMPLETE
    assert exc.value.observation.state is ObservationState.INCOMPLETE
    assert exc.value.observation.provider_attempt_count == 1
    assert exc.value.observation.transport_retry_count == 0
    assert exc.value.observation.conformance_retry_count == 0
    assert exc.value.observation.reasoning_tokens == 2
    assert exc.value.observation.input_tokens == 7
    assert exc.value.observation.cached_input_tokens == 0
    assert exc.value.observation.output_tokens == 3


@pytest.mark.asyncio
async def test_transport_and_conformance_retries_are_distinguishable() -> None:
    provider = ScriptedText(
        [
            _result(text="nope", parsed=None, input_tokens=1, output_tokens=1),
            ProviderError.from_code(FailureCode.RATE_LIMITED),
            _result(input_tokens=2, output_tokens=2),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(_request(deadline_ms=10_000))
    assert result.parsed == {"name": "ok", "count": 1}
    assert result.observation.conformance_retry_count == 1
    assert result.observation.retry_count == 1
    assert result.observation.transport_retry_count == 1
    assert result.observation.provider_attempt_count == 3
    assert result.observation.input_tokens is None
    assert result.observation.cached_input_tokens is None
    assert result.observation.output_tokens is None


@pytest.mark.asyncio
async def test_repair_inherits_reasoning_and_retry_ceiling_with_truthful_usage() -> None:
    provider = ScriptedText(
        [
            _result(text="not-json", parsed=None, reasoning_tokens=2),
            ProviderError.from_code(FailureCode.RATE_LIMITED),
            _result(reasoning_tokens=3),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(
        _request(
            reasoning_effort="high",
            max_transport_retries=1,
            deadline_ms=10_000,
        )
    )
    assert result.observation.conformance_retry_count == 1
    assert result.observation.transport_retry_count == 1
    assert result.observation.provider_attempt_count == 3
    assert result.observation.reasoning_tokens is None  # retry failure supplied no usage
    assert [call.reasoning_effort for call in provider.calls] == ["high"] * 3
    assert [call.max_transport_retries for call in provider.calls] == [1] * 3


@pytest.mark.asyncio
async def test_structured_reasoning_tokens_sum_or_remain_unknown() -> None:
    for second, expected in ((3, 5), (None, None)):
        provider = ScriptedText(
            [
                _result(text="not-json", parsed=None, reasoning_tokens=2),
                _result(reasoning_tokens=second),
            ]
        )
        result = await GenerationClient(text_provider=provider).generate_structured(
            _request(max_transport_retries=0)
        )
        assert result.observation.reasoning_tokens == expected
        assert result.observation.output_tokens == 8


@pytest.mark.asyncio
async def test_repair_preserves_explicit_none_temperature() -> None:
    provider = ScriptedText(
        [
            _result(text="not-json", parsed=None),
            _result(),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(_request(temperature=None))
    assert result.parsed == {"name": "ok", "count": 1}
    assert len(provider.calls) == 2
    assert provider.calls[0].temperature is None
    assert provider.calls[1].temperature is None
    assert result.observation.conformance_retry_count == 1


@pytest.mark.asyncio
async def test_repair_shares_overall_deadline(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _no_sleep(_delay: float) -> None:
        sleeps.append(_delay)

    monkeypatch.setattr("generationengine.client.asyncio.sleep", _no_sleep)
    provider = ScriptedText(
        [
            _result(text="bad", parsed=None),
            ProviderError.from_code(FailureCode.RATE_LIMITED),
            ProviderError.from_code(FailureCode.RATE_LIMITED),
            ProviderError.from_code(FailureCode.RATE_LIMITED),
        ]
    )
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_structured(_request(deadline_ms=80))
    assert exc.value.failure.code is FailureCode.RATE_LIMITED
    assert sleeps == []
    assert exc.value.observation.conformance_retry_count == 1


@pytest.mark.asyncio
async def test_repair_timeout_is_provider_timeout() -> None:
    import asyncio

    class SlowSecond(ScriptedText):
        async def generate(self, call: TextGenerationCall) -> TextGenerationResult:
            self.calls.append(call)
            if len(self.calls) == 1:
                return _result(text="bad", parsed=None)
            await asyncio.sleep(1)
            return _result()

    client = GenerationClient(text_provider=SlowSecond([]))
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_structured(_request(deadline_ms=50))
    assert exc.value.failure.code is FailureCode.PROVIDER_TIMEOUT
    assert exc.value.observation.conformance_retry_count == 1


@pytest.mark.asyncio
async def test_repair_refusal_is_provider_refused() -> None:
    provider = ScriptedText(
        [
            _result(text="bad", parsed=None),
            _result(refused=True, text=None, parsed=None),
        ]
    )
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_structured(_request())
    assert exc.value.failure.code is FailureCode.PROVIDER_REFUSED
    assert exc.value.observation.state is ObservationState.REFUSED
    assert exc.value.observation.conformance_retry_count == 1


@pytest.mark.asyncio
async def test_usage_unknown_propagates_instead_of_partial_sum() -> None:
    provider = ScriptedText(
        [
            _result(text="bad", parsed=None, input_tokens=10, output_tokens=1),
            _result(input_tokens=None, output_tokens=None, cached_input_tokens=None),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(_request())
    assert result.parsed == {"name": "ok", "count": 1}
    assert result.observation.input_tokens is None
    assert result.observation.output_tokens is None
    assert result.observation.cost_usd is None


@pytest.mark.asyncio
async def test_known_usage_is_summed_across_attempts() -> None:
    provider = ScriptedText(
        [
            _result(text="bad", parsed=None, input_tokens=4, cached_input_tokens=1, output_tokens=2),
            _result(input_tokens=6, cached_input_tokens=0, output_tokens=3),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(_request())
    assert result.observation.input_tokens == 10
    assert result.observation.cached_input_tokens == 1
    assert result.observation.output_tokens == 5
    assert result.observation.cost_usd is None


@pytest.mark.asyncio
async def test_provider_error_unknown_usage_does_not_keep_partial_total() -> None:
    provider = ScriptedText(
        [
            _result(text="bad", parsed=None, input_tokens=100, output_tokens=10),
            ProviderError.from_code(FailureCode.PROVIDER_ERROR),
        ]
    )
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_structured(_request())
    assert exc.value.failure.code is FailureCode.PROVIDER_ERROR
    assert exc.value.observation.input_tokens is None
    assert exc.value.observation.output_tokens is None
    assert exc.value.observation.cached_input_tokens is None
    assert exc.value.observation.conformance_retry_count == 1


@pytest.mark.asyncio
async def test_provider_error_known_usage_is_included_in_aggregate() -> None:
    provider = ScriptedText(
        [
            _result(text="bad", parsed=None, input_tokens=100, cached_input_tokens=2, output_tokens=10),
            ProviderError.from_code(
                FailureCode.PROVIDER_ERROR,
                input_tokens=25,
                cached_input_tokens=1,
                output_tokens=3,
            ),
        ]
    )
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_structured(_request())
    assert exc.value.failure.code is FailureCode.PROVIDER_ERROR
    assert exc.value.observation.input_tokens == 125
    assert exc.value.observation.cached_input_tokens == 3
    assert exc.value.observation.output_tokens == 13
    assert exc.value.observation.conformance_retry_count == 1


@pytest.mark.asyncio
async def test_recovered_retry_unknown_usage_makes_aggregate_none() -> None:
    provider = ScriptedText(
        [
            _result(text="nope", parsed=None, input_tokens=1, output_tokens=1),
            ProviderError.from_code(FailureCode.RATE_LIMITED),
            _result(input_tokens=2, output_tokens=2),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(_request(deadline_ms=10_000))
    assert result.parsed == {"name": "ok", "count": 1}
    assert result.observation.provider_attempt_count == 3
    assert result.observation.transport_retry_count == 1
    assert result.observation.conformance_retry_count == 1
    assert result.observation.input_tokens is None
    assert result.observation.cached_input_tokens is None
    assert result.observation.output_tokens is None
    assert result.observation.cost_usd is None


@pytest.mark.asyncio
async def test_recovered_retry_known_usage_is_included_in_aggregate() -> None:
    provider = ScriptedText(
        [
            _result(text="nope", parsed=None, input_tokens=1, cached_input_tokens=0, output_tokens=1),
            ProviderError.from_code(
                FailureCode.RATE_LIMITED,
                input_tokens=5,
                cached_input_tokens=0,
                output_tokens=0,
            ),
            _result(input_tokens=2, cached_input_tokens=0, output_tokens=2),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(_request(deadline_ms=10_000))
    assert result.parsed == {"name": "ok", "count": 1}
    assert result.observation.provider_attempt_count == 3
    assert result.observation.transport_retry_count == 1
    assert result.observation.conformance_retry_count == 1
    assert result.observation.input_tokens == 8
    assert result.observation.cached_input_tokens == 0
    assert result.observation.output_tokens == 3
    assert result.observation.cost_usd is None


@pytest.mark.asyncio
async def test_invalid_caller_schema_is_invalid_request_without_provider_call() -> None:
    provider = ScriptedText([])
    client = GenerationClient(text_provider=provider)
    with pytest.raises(GenerationEngineError) as exc:
        await client.generate_structured(
            _request(json_schema={"type": "not-a-schema-type"})
        )
    assert exc.value.failure.code is FailureCode.INVALID_REQUEST
    assert exc.value.observation.provider_attempt_count == 0
    assert exc.value.observation.retry_count == 0
    assert provider.calls == []


@pytest.mark.asyncio
async def test_correction_rate_limit_unknown_input_keeps_other_aggregates(
    monkeypatch,
) -> None:
    sleeps: list[float] = []

    async def _no_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("generationengine.client.asyncio.sleep", _no_sleep)
    provider = ScriptedText(
        [
            _result(
                text="nope",
                parsed=None,
                input_tokens=10,
                cached_input_tokens=0,
                output_tokens=1,
                provider_request_id="req-1",
                provider_response_id="resp-1",
            ),
            ProviderError.from_code(
                FailureCode.RATE_LIMITED,
                cached_input_tokens=0,
                output_tokens=0,
            ),
            _result(
                input_tokens=20,
                cached_input_tokens=0,
                output_tokens=2,
                provider_request_id="req-3",
                provider_response_id="resp-3",
            ),
        ]
    )
    client = GenerationClient(text_provider=provider)
    result = await client.generate_structured(_request(deadline_ms=10_000))
    assert result.parsed == {"name": "ok", "count": 1}
    assert result.observation.provider_attempt_count == 3
    assert result.observation.retry_count == 1
    assert result.observation.transport_retry_count == 1
    assert result.observation.conformance_retry_count == 1
    assert result.observation.input_tokens is None
    assert result.observation.cached_input_tokens == 0
    assert result.observation.output_tokens == 3
    assert result.observation.cost_usd is None
    assert result.observation.provider_request_id == "req-3"
    assert result.observation.provider_response_id == "resp-3"
    assert result.observation.response_model == "model"
    assert sleeps == [0.5]
    assert "," not in (result.observation.provider_request_id or "")
