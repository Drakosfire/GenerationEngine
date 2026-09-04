"""Provider-layer errors. Never exposed as the public consumer contract."""

from __future__ import annotations

from generationengine.failures import FailureCode, InferenceFailure, Retryability


class ProviderError(Exception):
    def __init__(
        self,
        failure: InferenceFailure,
        *,
        provider_request_id: str | None = None,
        response_model: str | None = None,
        input_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.provider_request_id = provider_request_id
        self.response_model = response_model
        self.input_tokens = input_tokens
        self.cached_input_tokens = cached_input_tokens
        self.output_tokens = output_tokens

    @property
    def retryable(self) -> bool:
        return self.failure.retryability is Retryability.YES

    @classmethod
    def from_code(cls, code: FailureCode, message: str, **kwargs: object) -> ProviderError:
        return cls(InferenceFailure.from_code(code, message), **kwargs)  # type: ignore[arg-type]
