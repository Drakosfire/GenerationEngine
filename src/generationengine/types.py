"""Public execution request/result types. No provider SDK types."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, FiniteFloat, model_validator

from generationengine.catalog import InferenceProfile
from generationengine.failures import InferenceFailure
from generationengine.observation import InferenceObservation


class GenerationEngineError(Exception):
    """Public failure. Carries normalized failure + observation; never an SDK type."""

    def __init__(self, failure: InferenceFailure, observation: InferenceObservation) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.observation = observation


class TextRequest(BaseModel):
    user_prompt: str
    system_prompt: str | None = None
    profile: InferenceProfile | None = None
    provider: str | None = None
    model: str | None = None
    temperature: float | None = 0.7
    reasoning_effort: str | None = Field(default=None, min_length=1)
    max_transport_retries: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of generated output tokens for this text operation.",
    )
    json_object: bool = Field(
        default=False,
        description="Request a provider-native JSON object without schema validation or parsing.",
    )
    json_schema: dict[str, Any] | None = None
    schema_name: str | None = None
    deadline_ms: int | None = Field(
        default=None,
        ge=1,
        description="Overall GenerationEngine operation budget in milliseconds, including retries.",
    )


class TextResult(BaseModel):
    text: str | None = None
    parsed: dict[str, Any] | None = None
    observation: InferenceObservation


class BinaryDecisionQuestion(BaseModel):
    kind: Literal["binary"] = "binary"
    name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    true_description: str | None = None
    false_description: str | None = None


def _validate_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    if len(labels) < 2 or any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Decision labels must contain at least two distinct nonblank values")
    return labels


class ChoiceDecisionQuestion(BaseModel):
    kind: Literal["choice"] = "choice"
    name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: tuple[str, ...]
    option_descriptions: dict[str, str | None] | None = None

    @model_validator(mode="after")
    def validate_options(self) -> ChoiceDecisionQuestion:
        _validate_labels(self.options)
        if self.option_descriptions is not None and (
            set(self.option_descriptions) != set(self.options)
            or any(value is not None and not value.strip() for value in self.option_descriptions.values())
        ):
            raise ValueError("Choice descriptions must cover exactly the named options")
        return self


class ScoreDecisionQuestion(BaseModel):
    kind: Literal["score"] = "score"
    name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    levels: tuple[str, ...]

    @model_validator(mode="after")
    def validate_levels(self) -> ScoreDecisionQuestion:
        _validate_labels(self.levels)
        return self


DecisionQuestion = Annotated[
    BinaryDecisionQuestion | ChoiceDecisionQuestion | ScoreDecisionQuestion,
    Field(discriminator="kind"),
]


class DecisionRequest(BaseModel):
    state: Any
    questions: tuple[DecisionQuestion, ...] = Field(min_length=1)
    provider: str | None = None
    model: str | None = None
    max_transport_retries: int | None = Field(default=None, ge=0)
    deadline_ms: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_request(self) -> DecisionRequest:
        names = [question.name for question in self.questions]
        if len(names) != len(set(names)):
            raise ValueError("Decision question names must be unique")
        try:
            encoded = json.dumps(self.state, allow_nan=False)
            if json.loads(encoded) != self.state:
                raise ValueError("Decision state changes under JSON encoding")
        except (TypeError, ValueError) as exc:
            raise ValueError("Decision state must be JSON-safe") from exc
        return self


class BinaryDecisionAnswer(BaseModel):
    kind: Literal["binary"] = "binary"
    value: bool
    probability_true: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class ChoiceDecisionAnswer(BaseModel):
    kind: Literal["choice"] = "choice"
    selected: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class ScoreDecisionAnswer(BaseModel):
    kind: Literal["score"] = "score"
    score: FiniteFloat
    distribution: dict[str, float] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


DecisionAnswer = Annotated[
    BinaryDecisionAnswer | ChoiceDecisionAnswer | ScoreDecisionAnswer,
    Field(discriminator="kind"),
]


class DecisionResult(BaseModel):
    answers: dict[str, DecisionAnswer]
    observation: InferenceObservation


class ImageRequest(BaseModel):
    prompt: str
    profile: InferenceProfile | None = None
    model: str | None = None
    num_images: int = Field(default=1, ge=1, le=8)
    width: int = Field(default=1024, ge=1)
    height: int = Field(default=1024, ge=1)
    negative_prompt: str | None = None
    source_image_url: str | None = None
    source_image_bytes: bytes | None = None
    mask_base64: str | None = None
    base_image_base64: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    deadline_ms: int | None = Field(
        default=None,
        ge=1,
        description="Overall GenerationEngine operation budget in milliseconds, including retries.",
    )


class GeneratedImage(BaseModel):
    content: bytes
    media_type: str = "image/png"
    width: int | None = None
    height: int | None = None


class ImageResult(BaseModel):
    images: list[GeneratedImage]
    observation: InferenceObservation
