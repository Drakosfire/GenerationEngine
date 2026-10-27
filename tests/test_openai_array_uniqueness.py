"""OpenAI wire projection must not weaken caller-side array conformance."""
from __future__ import annotations

import copy
import json

import httpx
import pytest
from openai import AsyncOpenAI

from generationengine import FailureCode, GenerationClient, GenerationEngineError, TextRequest
from generationengine.providers.openai_text import OpenAITextProvider
from generationengine.utils.schema_utils import make_schema_strict

SCHEMA = {
    "type": "object",
    "properties": {
        "observation_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_refs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_quote_ids": {
                                    "type": "array", "items": {"type": "string"},
                                    "minItems": 1, "uniqueItems": True,
                                }
                            },
                            "required": ["source_quote_ids"], "additionalProperties": False,
                        },
                    }
                },
                "required": ["evidence_refs"], "additionalProperties": False,
            },
        }
    },
    "required": ["observation_nodes"], "additionalProperties": False,
}


def candidate(ids):
    return {"observation_nodes": [{"evidence_refs": [{"source_quote_ids": ids}]}]}


def test_nested_projection_preserves_original_schema_and_other_constraints():
    original = copy.deepcopy(SCHEMA)
    projected = make_schema_strict(SCHEMA)
    ids = projected["properties"]["observation_nodes"]["items"]["properties"]["evidence_refs"]["items"]["properties"]["source_quote_ids"]
    assert ids == {"type": "array", "items": {"type": "string"}, "minItems": 1}
    assert SCHEMA == original


def test_projection_visits_definitions_and_compositions_but_not_property_names_or_literals():
    literal = {"uniqueItems": True, "other": [1, 1]}
    schema = {
        "type": "object",
        "properties": {
            "uniqueItems": {"type": "boolean"},
            "payload": {"type": "string", "examples": [literal], "default": literal,
                        "enum": [literal], "const": literal},
            "refs": {"anyOf": [{"type": "array", "items": {"type": "string"},
                                  "uniqueItems": True}, {"type": "null"}]},
        },
        "$defs": {"ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
    }
    projected = make_schema_strict(schema)
    assert projected["properties"]["uniqueItems"] == {"type": "boolean"}
    assert projected["properties"]["payload"] == schema["properties"]["payload"]
    assert "uniqueItems" not in projected["$defs"]["ids"]
    assert "uniqueItems" not in projected["properties"]["refs"]
    assert schema["$defs"]["ids"]["uniqueItems"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [False, True])
async def test_real_sdk_wire_projection_and_original_local_validation(repair):
    calls = []
    outputs = [candidate(["q1", "q1"]), candidate(["q1"])]

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        text = json.dumps(outputs[len(calls) - 1])
        return httpx.Response(200, headers={"x-request-id": f"req-{len(calls)}"}, json={
            "id": f"resp-{len(calls)}", "object": "response", "created_at": 1,
            "status": "completed", "model": "gpt-6-luna", "error": None,
            "output": [{"id": "msg-1", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                        "text": text, "annotations": []}]}],
        })

    sdk = AsyncOpenAI(api_key="synthetic-test-key", max_retries=0,
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    engine = GenerationClient(text_provider=OpenAITextProvider(client=sdk))
    original = copy.deepcopy(SCHEMA)
    request = TextRequest(provider="openai", model="gpt-6-luna", user_prompt="Return evidence references.",
                          json_schema=SCHEMA, temperature=None, max_conformance_retries=int(repair))
    try:
        if repair:
            result = await engine.generate_structured(request)
            assert result.parsed == candidate(["q1"])
            assert result.observation.conformance_retry_count == 1
            assert result.observation.provider_attempt_count == 2
            assert result.observation.transport_retry_count == 0
        else:
            with pytest.raises(GenerationEngineError) as caught:
                await engine.generate_structured(request)
            assert caught.value.failure.code == FailureCode.STRUCTURED_OUTPUT_INVALID
        assert len(calls) == (2 if repair else 1)
        for body in calls:
            wire = body["text"]["format"]
            assert wire["strict"] is True
            ids = wire["schema"]["properties"]["observation_nodes"]["items"]["properties"]["evidence_refs"]["items"]["properties"]["source_quote_ids"]
            assert "uniqueItems" not in ids
            assert ids["minItems"] == 1
        assert SCHEMA == original
        assert request.json_schema == original
    finally:
        await engine.aclose()
