"""GEJ-02: official SDK mapping remains behind the decision provider boundary."""

from __future__ import annotations

import httpx2
import pytest
from typesafe_sdk import (
    Choice,
    Noul,
    Score,
    SystemOneResponse,
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)

from generationengine import (
    BinaryDecisionQuestion,
    ChoiceDecisionQuestion,
    DecisionRequest,
    FailureCode,
    GenerationClient,
    GenerationEngineError,
    ScoreDecisionQuestion,
)
from generationengine.providers.typesafe_decision import (
    GATEWAY_BASE_URL,
    TRANSPORT,
    TypeSafeDecisionProvider,
)


def request() -> DecisionRequest:
    return DecisionRequest(
        state={"message": "A sample text"},
        questions=(
            BinaryDecisionQuestion(
                name="yes", question="Is it relevant?",
                true_description="Relevant", false_description="Irrelevant",
            ),
            ChoiceDecisionQuestion(
                name="category", question="Which category?", options=("a", "b")
            ),
            ScoreDecisionQuestion(
                name="strength", question="How strong?", levels=("low", "medium", "high")
            ),
        ),
        provider="typesafe",
        model="typesafe-ai/jev",
        max_transport_retries=0,
    )


def sdk_response(*, model: str = "typesafe-ai/jev") -> SystemOneResponse:
    return SystemOneResponse.model_validate({
        "model": model,
        "usage": {"input_tokens": 14, "output_tokens": 0},
        "answers": {
            "yes": {"type": "noul", "noul": 0.8},
            "category": {"type": "choice", "choice": "a", "confidence": 0.7,
                         "probabilities": {"a": 0.7, "b": 0.3}},
            "strength": {"type": "score", "score": 1.4, "confidence": 0.6,
                         "legend": {0: "low", 1: "medium", 2: "high"},
                         "probabilities": {0: 0.1, 1: 0.4, 2: 0.5}},
        },
    })


class FakeSDKClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def system_one(self, state, questions, *, model):
        self.calls.append((state, questions, model))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.mark.asyncio
async def test_three_question_families_and_truthful_gateway_observation():
    sdk = FakeSDKClient(sdk_response())
    provider = TypeSafeDecisionProvider(client=sdk)
    result = await GenerationClient(decision_providers={"typesafe": provider}).decide(request())
    state, questions, model = sdk.calls[0]
    assert state == {"message": "A sample text"}
    assert model == "typesafe-ai/jev"
    assert isinstance(questions["yes"], Noul)
    assert questions["yes"].criteria == {"true": "Relevant", "false": "Irrelevant"}
    assert isinstance(questions["category"], Choice)
    assert list(questions["category"].criteria) == ["a", "b"]
    assert isinstance(questions["strength"], Score)
    assert questions["strength"].criteria == ["low", "medium", "high"]
    assert result.answers["yes"].value is True
    assert result.answers["yes"].probability_true == 0.8
    assert result.answers["category"].selected == "a"
    assert result.answers["strength"].score == 1.4
    assert result.answers["strength"].distribution == {"low": 0.1, "medium": 0.4, "high": 0.5}
    assert result.observation.provider == "typesafe"
    assert result.observation.provider_transport == TRANSPORT
    assert result.observation.requested_model == "typesafe-ai/jev"
    assert result.observation.resolved_model == "typesafe-ai/jev"
    assert result.observation.response_model == "typesafe-ai/jev"
    assert result.observation.input_tokens == 14
    assert result.observation.output_tokens == 0
    assert result.observation.provider_attempt_count == 1


@pytest.mark.asyncio
async def test_missing_key_is_config_failure_before_sdk_attempt(monkeypatch):
    monkeypatch.delenv("TYPESAFE_JEV_API_KEY", raising=False)
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient.from_env().decide(request())
    assert exc.value.failure.code is FailureCode.CONFIGURATION_UNAVAILABLE
    assert exc.value.observation.provider_attempt_count == 0


@pytest.mark.asyncio
async def test_lazy_provider_passes_project_key_and_disables_sdk_retries(monkeypatch):
    import generationengine.providers.typesafe_decision as adapter

    configured = {}

    class OwnedClient(FakeSDKClient):
        def __init__(self, **kwargs):
            configured.update(kwargs)
            super().__init__(sdk_response())

        async def aclose(self):
            configured["closed"] = True

    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", "private-project-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "unrelated-upstream-key")
    monkeypatch.setattr(adapter, "AsyncTypeSafeClient", OwnedClient)
    client = GenerationClient.from_env()
    result = await client.decide(request())
    assert result.observation.provider_transport == TRANSPORT
    assert configured["api_key"] == "private-project-key"
    assert configured["base_url"] == GATEWAY_BASE_URL
    assert configured["retry"].max_retries == 0
    assert "private-project-key" not in result.model_dump_json()
    await client.aclose()
    assert configured["closed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TypeSafeAuthenticationError(401, {}, httpx2.Headers()), FailureCode.CONFIGURATION_UNAVAILABLE),
        (TypeSafeRateLimitError(429, {}, httpx2.Headers()), FailureCode.RATE_LIMITED),
    ],
)
async def test_sdk_failures_are_normalized(error, code):
    provider = TypeSafeDecisionProvider(client=FakeSDKClient(error))
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient(decision_providers={"typesafe": provider}).decide(request())
    assert exc.value.failure.code is code
    assert exc.value.observation.provider_transport == TRANSPORT
    assert exc.value.observation.provider_attempt_count == 1
    assert "typesafe_sdk" not in exc.value.failure.message


@pytest.mark.asyncio
async def test_malformed_score_legend_fails_closed():
    response = sdk_response()
    bad = response.model_copy(update={"answers": {
        **response.answers,
        "strength": response.answers["strength"].model_copy(update={"legend": {0: "wrong"}}),
    }})
    provider = TypeSafeDecisionProvider(client=FakeSDKClient(bad))
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient(decision_providers={"typesafe": provider}).decide(request())
    assert exc.value.failure.code is FailureCode.MALFORMED_PROVIDER_RESPONSE


@pytest.mark.asyncio
async def test_provider_rejects_sdk_unsupported_scalar_state():
    sdk = FakeSDKClient(sdk_response())
    provider = TypeSafeDecisionProvider(client=sdk)
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient(decision_providers={"typesafe": provider}).decide(
            request().model_copy(update={"state": 7})
        )
    assert exc.value.failure.code is FailureCode.INVALID_REQUEST
    assert sdk.calls == []
