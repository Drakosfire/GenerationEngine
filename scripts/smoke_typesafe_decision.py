"""Opt-in live TypeSafe decision witness; requires TYPESAFE_JEV_API_KEY."""

from __future__ import annotations

import asyncio
import json

from generationengine import (
    BinaryDecisionQuestion,
    ChoiceDecisionQuestion,
    DecisionRequest,
    GenerationClient,
    GenerationEngineError,
    ScoreDecisionQuestion,
)


async def main() -> None:
    client = GenerationClient.from_env()
    try:
        result = await client.decide(DecisionRequest(
            state={"text": "A lantern gives bright light."},
            questions=(
                BinaryDecisionQuestion(name="light", question="Does the text mention light?"),
                ChoiceDecisionQuestion(name="kind", question="What is mentioned?",
                                       options=("lantern", "sword")),
                ScoreDecisionQuestion(name="clarity", question="How explicit is the statement?",
                                      levels=("absent", "implicit", "explicit")),
            ),
            provider="typesafe",
            model="typesafe-ai/jev",
            max_transport_retries=0,
        ))
        observation = result.observation
        print(json.dumps({
            "provider": observation.provider,
            "provider_transport": observation.provider_transport,
            "requested_model": observation.requested_model,
            "resolved_model": observation.resolved_model,
            "response_model": observation.response_model,
            "input_tokens": observation.input_tokens,
            "output_tokens": observation.output_tokens,
            "answers": {name: answer.model_dump(mode="json")
                        for name, answer in result.answers.items()},
        }, sort_keys=True))
    except GenerationEngineError as exc:
        print(json.dumps({"failure_code": exc.failure.code.value,
                          "provider_transport": exc.observation.provider_transport}))
        raise SystemExit(1) from None
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
