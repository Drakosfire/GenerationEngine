"""GEJ-01 public typed-decision contract with a provider-neutral fake."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from generationengine import (
    BinaryDecisionQuestion,
    Capability,
    ChoiceDecisionQuestion,
    DecisionGenerationResult,
    DecisionRequest,
    FailureCode,
    GenerationClient,
    GenerationEngineError,
    ObservationState,
    ScoreDecisionQuestion,
)
from generationengine.providers.errors import ProviderError
from generationengine.resolver import LIVE_MODELS, ResolutionError, resolve


class FakeDecisionProvider:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def decide(self, call):
        self.calls.append(call)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def questions():
    return (
        BinaryDecisionQuestion(name="ready", question="Is this ready?",
                               true_description="Ready", false_description="Not ready"),
        ChoiceDecisionQuestion(name="route", question="Which route?", options=("north", "south")),
        ScoreDecisionQuestion(name="quality", question="How strong?", levels=("low", "medium", "high")),
    )


def request(**overrides):
    return DecisionRequest(state={"count": 2}, questions=questions(),
                           provider="example", model="floating-alias", **overrides)


def completed_result(**overrides):
    payload = {
        "answers": {
            "ready": {"kind": "binary", "value": True, "probability_true": 0.8},
            "route": {"kind": "choice", "selected": "north", "probabilities": {"north": 0.7, "south": 0.3}},
            "quality": {"kind": "score", "score": 2.4, "distribution": {"low": 0.1, "medium": 0.4, "high": 0.5}},
        },
        "provider_request_id": "req-1",
        "provider_transport": "gateway",
        "input_tokens": 0,
        "output_tokens": 3,
    }
    payload.update(overrides)
    return DecisionGenerationResult.model_validate(payload)


def test_binary_choice_score_validation_and_json_state():
    with pytest.raises(ValidationError):
        ScoreDecisionQuestion(name="x", question="Score?", levels=("only",))
    with pytest.raises(ValidationError):
        ChoiceDecisionQuestion(name="x", question="Choose?", options=("a", "a"))
    with pytest.raises(ValidationError):
        ChoiceDecisionQuestion(name="x", question="Choose?", options=("a", "b"),
                               option_descriptions={"a": "one"})
    described = ChoiceDecisionQuestion(name="x", question="Choose?", options=("a", "b"),
                                       option_descriptions={"a": "one", "b": None})
    assert described.option_descriptions == {"a": "one", "b": None}
    with pytest.raises(ValidationError):
        DecisionRequest(state={"bad": float("nan")}, questions=questions(),
                        provider="example", model="x")
    with pytest.raises(ValidationError):
        DecisionRequest(state={1: "coerced-key"}, questions=questions(),
                        provider="example", model="x")
    with pytest.raises(ValidationError):
        DecisionRequest(state={}, questions=(questions()[0], questions()[0]),
                        provider="example", model="x")


@pytest.mark.asyncio
async def test_invalid_request_is_ge_invalid_request_before_provider():
    fake = FakeDecisionProvider()
    client = GenerationClient(decision_providers={"example": fake})
    with pytest.raises(GenerationEngineError) as exc:
        await client.decide({"state": {}, "questions": [{"kind": "score", "name": "x",
                                                       "question": "Score?", "levels": ["only"]}],
                             "provider": "example", "model": "x"})
    assert exc.value.failure.code is FailureCode.INVALID_REQUEST
    assert exc.value.observation.provider_attempt_count == 0
    assert fake.calls == []


@pytest.mark.asyncio
async def test_explicit_decision_target_and_truthful_observation():
    assert "floating-alias" not in LIVE_MODELS
    resolution = resolve(capability=Capability.DECISION, provider="example",
                         model="floating-alias", registered_providers=frozenset({"example"}))
    assert resolution.resolved_model == "floating-alias"
    fake = FakeDecisionProvider(completed_result())
    result = await GenerationClient(decision_providers={"example": fake}).decide(request())
    assert fake.calls[0].model == "floating-alias"
    assert result.answers["ready"].value is True
    assert result.answers["route"].selected == "north"
    assert result.answers["quality"].score == 2.4
    assert result.observation.provider == "example"
    assert result.observation.requested_model == "floating-alias"
    assert result.observation.resolved_model == "floating-alias"
    assert result.observation.response_model is None  # no invented upstream version
    assert result.observation.provider_transport == "gateway"
    assert result.observation.input_tokens == 0
    assert result.observation.cost_usd is None
    assert result.observation.provider_attempt_count == 1
    assert result.observation.state is ObservationState.COMPLETED


@pytest.mark.asyncio
async def test_missing_capability_and_model_fail_without_text_fallback():
    fake = FakeDecisionProvider()
    client = GenerationClient(decision_providers={"example": fake})
    with pytest.raises(GenerationEngineError) as exc:
        await client.decide(DecisionRequest(state={}, questions=questions(), provider="example"))
    assert exc.value.failure.code is FailureCode.INVALID_REQUEST
    with pytest.raises(GenerationEngineError) as exc:
        await client.decide(DecisionRequest(state={}, questions=questions(),
                                            provider="openai", model="gpt-5.1"))
    assert exc.value.failure.code is FailureCode.UNSUPPORTED_CAPABILITY
    assert fake.calls == []
    with pytest.raises(ResolutionError):
        resolve(capability=Capability.DECISION, provider="openai", model="gpt-5.1")


@pytest.mark.asyncio
async def test_provider_failure_is_normalized_with_one_observation():
    fake = FakeDecisionProvider(ProviderError.from_code(
        FailureCode.RATE_LIMITED, provider_request_id="retry-1",
        response_model="floating-alias", provider_transport="gateway"))
    client = GenerationClient(decision_providers={"example": fake})
    with pytest.raises(GenerationEngineError) as exc:
        await client.decide(request(max_transport_retries=0))
    assert exc.value.failure.code is FailureCode.RATE_LIMITED
    assert exc.value.observation.provider_attempt_count == 1
    assert exc.value.observation.response_model == "floating-alias"
    assert exc.value.observation.provider_transport == "gateway"
    assert exc.value.observation.state is ObservationState.FAILED


@pytest.mark.asyncio
async def test_retry_and_malformed_answer_use_existing_public_failures(monkeypatch):
    import generationengine.client as client_module

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(client_module.asyncio, "sleep", no_sleep)
    fake = FakeDecisionProvider(ProviderError.from_code(FailureCode.PROVIDER_UNAVAILABLE),
                                completed_result())
    result = await GenerationClient(decision_providers={"example": fake}).decide(request())
    assert len(fake.calls) == 2
    assert result.observation.retry_count == 1
    assert result.observation.provider_attempt_count == 2

    bad = completed_result()
    bad.answers["route"].selected = "outside-options"
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient(decision_providers={"example": FakeDecisionProvider(bad)}).decide(request())
    assert exc.value.failure.code is FailureCode.MALFORMED_PROVIDER_RESPONSE
    assert exc.value.observation.provider_attempt_count == 1


@pytest.mark.asyncio
async def test_timeout_and_unexpected_adapter_error_are_normalized():
    for error, code in ((TimeoutError(), FailureCode.PROVIDER_TIMEOUT),
                        (RuntimeError("secret SDK detail"), FailureCode.PROVIDER_ERROR)):
        fake = FakeDecisionProvider(error)
        with pytest.raises(GenerationEngineError) as exc:
            await GenerationClient(decision_providers={"example": fake}).decide(
                request(max_transport_retries=0)
            )
        assert exc.value.failure.code is code
        assert "secret SDK detail" not in exc.value.failure.message
        assert exc.value.observation.provider_attempt_count == 1
