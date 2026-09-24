# E5J — schema-less JSON-object mode

**Status:** ACTIVE
**Architecture owner:** `Drakosfire/DungeonOverMind`
**Execution repository:** `Drakosfire/GenerationEngine`
**Design authority:** attached canonical handoff `HANDOFF-E5J-generationengine-json-object-mode.md`
**Design/activation base:** `c5052be93cfbac99a89576648bf5c1c261d73835`
**PR topology:** `serial`
**Assigned branch:** `codex/e5j-json-object-mode`
**Assigned PR title:** `E5J: add schema-less JSON-object mode`

## Mission

Add `json_object: bool = False` to `TextRequest` and
`TextGenerationCall`. False/omitted preserves exact existing behavior. True asks
the provider for a valid JSON object but returns raw text with `parsed=None`:
GenerationEngine performs no parsing, schema validation, or conformance repair.

This is a GenerationEngine prerequisite contract only. No consumer repository
or Buddy document-planner migration is authorized.

## Public and provider contract

`json_object=True` is valid only for ordinary `generate_text()` without a JSON
Schema. `json_object=True` plus `json_schema` fails before provider execution
with existing `INVALID_REQUEST` semantics. `stream_text()` with JSON-object mode
also fails before provider execution; this slice adds no incremental JSON
assembly or local buffering.

Provider adapters own wire translation:

```text
OpenAI Responses  → text.format = {"type": "json_object"}
OpenRouter Chat   → response_format = {"type": "json_object"}
```

False omits those wire controls. Temperature, output ceiling, prompts, model,
retry, deadline, IDs, failures, and observations continue unchanged.

## Write lease

Expected modified paths:

```text
src/generationengine/types.py
src/generationengine/providers/base.py
src/generationengine/client.py
src/generationengine/providers/openai_text.py
src/generationengine/providers/openrouter_text.py
tests/test_client.py
tests/test_openai_text.py
tests/test_openrouter_text.py
docs/CORE-CONTRACT.md
```

`docs/CURRENT-STATE.md` is optional. Do not modify observation, catalog,
resolver, failures, dependency/version files, structured-conformance policy, or
any consumer repository.

## Merge-blocking witnesses

- Both public/core fields default to false.
- False preserves existing ordinary provider request shapes.
- True reaches the provider-neutral call unchanged.
- OpenAI and OpenRouter map true to their exact native fields and omit false.
- Successful and malformed JSON remain raw `TextResult.text` with `parsed=None`.
- No local parsing, repair, or structured routing occurs.
- Refusal/failure normalization remains unchanged.
- Transport retries preserve true.
- JSON-object plus schema rejects with `INVALID_REQUEST` and zero provider calls.
- Streaming JSON-object rejects with `INVALID_REQUEST` and zero provider calls.
- Temperature and `max_output_tokens` compose normally.
- Structured-conformance regression remains green.
- No consumer repository changes.

## Verification

```bash
uv lock --check
uv sync
uv run ruff check src/generationengine/types.py src/generationengine/providers/base.py src/generationengine/client.py src/generationengine/providers/openai_text.py src/generationengine/providers/openrouter_text.py tests/test_client.py tests/test_openai_text.py tests/test_openrouter_text.py
uv run pytest -q tests/test_client.py tests/test_openai_text.py tests/test_openrouter_text.py tests/test_structured_conformance.py tests/test_explicit_targets.py
uv run pytest -q
git diff --check
```

No live provider call is required.

## State sync after merge

Record the exact PR, reviewed head, merge SHA, review-cycle count, and evidence
in this handoff and DungeonOverMind authority. Only after E5J settles may a
separate Buddy document-planner migration be designed.

## Stop conditions

Stop rather than widen scope if main changed the text/adapter contract, another
GE PR overlaps the lease, the locked SDK cannot express either wire field,
provider-specific branching would leak into core, raw-text/no-repair semantics
require structured redesign, or a consumer change is required for proof.
