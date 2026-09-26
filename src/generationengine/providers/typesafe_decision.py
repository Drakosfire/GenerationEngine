"""TypeSafe decision execution through the current Vercel AI Gateway route."""

from __future__ import annotations

import os
from typing import Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from generationengine.failures import FailureCode
from generationengine.providers.base import DecisionGenerationCall, DecisionGenerationResult
from generationengine.providers.errors import ProviderError
from generationengine.types import (
    BinaryDecisionAnswer,
    BinaryDecisionQuestion,
    ChoiceDecisionAnswer,
    ChoiceDecisionQuestion,
    ScoreDecisionAnswer,
    ScoreDecisionQuestion,
)

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
TRANSPORT = "vercel_ai_gateway"


class TypeSafeDecisionProvider:
    """Keep SDK questions, credentials, transport and failures behind this adapter."""

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            self._owns_client = False
            return
        key = api_key if api_key is not None else os.getenv("TYPESAFE_JEV_API_KEY")
        if not key or not key.strip():
            raise ProviderError.from_code(
                FailureCode.CONFIGURATION_UNAVAILABLE,
                "TYPESAFE_JEV_API_KEY is required for TypeSafe decisions.",
                provider_transport=TRANSPORT,
            )
        try:
            # GE owns retries. The SDK must make exactly one HTTP attempt per call.
            self._client = AsyncTypeSafeClient(
                api_key=key, base_url=GATEWAY_BASE_URL, retry=RetryPolicy(max_retries=0)
            )
        except TypeSafeError as exc:
            raise ProviderError.from_code(
                FailureCode.CONFIGURATION_UNAVAILABLE,
                "TypeSafe decision provider configuration is invalid.",
                provider_transport=TRANSPORT,
            ) from exc
        self._owns_client = True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def decide(self, call: DecisionGenerationCall) -> DecisionGenerationResult:
        if not isinstance(call.state, (str, dict, list)):
            raise ProviderError.from_code(
                FailureCode.INVALID_REQUEST,
                "TypeSafe decisions require text, object, or array state.",
                provider_transport=TRANSPORT,
            )
        questions = _questions(call)
        try:
            response = await self._client.system_one(
                call.state, questions, model=call.model
            )
        except Exception as exc:
            raise _map_exception(exc) from exc
        metadata = _response_metadata(response)
        try:
            answers = _answers(call, response)
            usage = response.usage
            if metadata["response_model"] is None:
                raise ValueError("Missing response model")
            if usage is None or metadata["input_tokens"] is None or metadata["output_tokens"] is None:
                raise ValueError("Invalid response usage")
            return DecisionGenerationResult(
                answers=answers,
                **metadata,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError.from_code(
                FailureCode.MALFORMED_PROVIDER_RESPONSE,
                "TypeSafe returned an invalid decision response.",
                **metadata,
            ) from exc


def _questions(call: DecisionGenerationCall) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for question in call.questions:
        if isinstance(question, BinaryDecisionQuestion):
            criteria = {
                key: value for key, value in (
                    ("true", question.true_description),
                    ("false", question.false_description),
                ) if value is not None
            }
            questions[question.name] = Noul(
                instructions=question.question, criteria=criteria or None
            )
        elif isinstance(question, ChoiceDecisionQuestion):
            questions[question.name] = Choice(
                instructions=question.question,
                criteria={
                    label: question.option_descriptions[label]
                    if question.option_descriptions is not None else None
                    for label in question.options
                },
            )
        elif isinstance(question, ScoreDecisionQuestion):
            questions[question.name] = Score(
                instructions=question.question, criteria=list(question.levels)
            )
    return questions


def _answers(call: DecisionGenerationCall, response: Any) -> dict[str, Any]:
    if set(response.answers) != {question.name for question in call.questions}:
        raise ValueError("Answer names differ from questions")
    answers: dict[str, Any] = {}
    for question in call.questions:
        answer = response.answers[question.name]
        if isinstance(question, BinaryDecisionQuestion) and answer.type == "noul":
            answers[question.name] = BinaryDecisionAnswer(
                value=answer.noul >= 0.5, probability_true=answer.noul
            )
        elif isinstance(question, ChoiceDecisionQuestion) and answer.type == "choice":
            answers[question.name] = ChoiceDecisionAnswer(
                selected=answer.choice,
                probabilities=answer.probabilities,
                confidence=answer.confidence,
            )
        elif isinstance(question, ScoreDecisionQuestion) and answer.type == "score":
            indices = set(range(len(question.levels)))
            if set(answer.legend) != indices or set(answer.probabilities) != indices or any(
                answer.legend[index] != level for index, level in enumerate(question.levels)
            ):
                raise ValueError("Score legend differs from requested rubric")
            answers[question.name] = ScoreDecisionAnswer(
                score=answer.score,
                distribution={question.levels[index]: value for index, value in answer.probabilities.items()},
                confidence=answer.confidence,
            )
        else:
            raise ValueError("Answer type differs from question type")
    return answers


def _request_id(response: Any) -> str | None:
    try:
        request_id = response.request_id
        return request_id if isinstance(request_id, str) and request_id.strip() else None
    except (AttributeError, TypeSafeError):
        return None


def _response_metadata(response: Any) -> dict[str, Any]:
    """Retain only validated observation fields, never answer or response bodies."""
    model = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    return {
        "provider_request_id": _request_id(response),
        "response_model": model if isinstance(model, str) and model.strip() else None,
        "provider_transport": TRANSPORT,
        "input_tokens": input_tokens if type(input_tokens) is int and input_tokens >= 0 else None,
        "output_tokens": output_tokens if type(output_tokens) is int and output_tokens >= 0 else None,
    }


def _map_exception(exc: Exception) -> ProviderError:
    request_id = exc.request_id if isinstance(exc, TypeSafeAPIError) else None
    fields = {"provider_request_id": request_id, "provider_transport": TRANSPORT}
    if isinstance(exc, (TypeSafeAuthenticationError, TypeSafePermissionDeniedError)):
        return ProviderError.from_code(
            FailureCode.CONFIGURATION_UNAVAILABLE,
            "TypeSafe rejected the configured credential.", **fields
        )
    if isinstance(exc, TypeSafeRateLimitError):
        return ProviderError.from_code(FailureCode.RATE_LIMITED, **fields)
    if isinstance(exc, TypeSafeAPITimeoutError):
        return ProviderError.from_code(FailureCode.PROVIDER_TIMEOUT, **fields)
    if isinstance(exc, TypeSafeAPIResponseValidationError):
        return ProviderError.from_code(
            FailureCode.MALFORMED_PROVIDER_RESPONSE,
            "TypeSafe response failed SDK validation.", **fields
        )
    if isinstance(exc, (TypeSafeBadRequestError, TypeSafeUnprocessableEntityError)):
        return ProviderError.from_code(
            FailureCode.INVALID_REQUEST, "TypeSafe rejected the decision request.", **fields
        )
    if isinstance(exc, (TypeSafeInternalServerError, TypeSafeAPIConnectionError)):
        return ProviderError.from_code(FailureCode.PROVIDER_UNAVAILABLE, **fields)
    if isinstance(exc, TypeSafeError):
        return ProviderError.from_code(FailureCode.PROVIDER_ERROR, **fields)
    return ProviderError.from_code(FailureCode.PROVIDER_ERROR, **fields)
