# Structured conformance

**Status:** ACTIVE ADOPTED DECISION — local validation and one bounded structural repair  
**Date:** 2026-09-16  
**Related:** `CORE-CONTRACT.md` §8 Structured output

## Contract

GenerationEngine owns structural conformance of model output to a caller-supplied schema.

The caller owns the schema and its domain meaning. GenerationEngine owns obtaining a response that conforms to that schema or returning a normalized structured-output failure.

```text
caller
  supplies prompt + JSON Schema
        ↓
GenerationEngine.generate_structured
        ↓
provider-specific structured strategy
        ↓
parse
        ↓
local schema validation
        ↓
valid? ── yes ──→ return parsed schema-conforming object
  │
  no
  ↓
bounded corrective inference retry
  ↓
valid result OR STRUCTURED_OUTPUT_INVALID
```

The public semantic promise is conformance, not use of any particular provider-native structured-output feature.

## Ownership boundary

GenerationEngine owns:

- JSON/text acquisition for the structured operation;
- provider-native structured-output mechanics where available;
- parsing generated JSON;
- JSON Schema validation;
- generic structural validation feedback;
- bounded corrective inference retries for structural mismatches;
- normalized failure when conformance cannot be achieved;
- operation-level timing and inference accounting for the attempts it owns.

Consumers own:

- schema definition;
- prompt/task meaning;
- domain/business validation;
- interpretation of a structurally valid result;
- workflow retries for reasons other than inference conformance.

Examples of GE-owned structural failure:

```text
invalid JSON
missing required field
wrong type
invalid enum
forbidden extra property
nested schema mismatch
```

Examples that remain consumer-owned:

```text
claim unsupported by source evidence
world-history contradiction
illegal publication transition
application/business invariant violation
semantically implausible domain result
```

A consumer may still instantiate its own Pydantic/domain model from `result.parsed`. That construction is a product boundary/type concern; it should not need to reimplement generic model-repair loops for structural mismatches already covered by the supplied schema.

## Provider strategy

Provider-native strict-schema support is an optimization, not the contract.

GenerationEngine may choose among strategies such as:

```text
native strict JSON Schema
→ local schema validation

JSON-object response mode
→ local schema validation
→ corrective retry if invalid

ordinary text response
→ JSON parse
→ local schema validation
→ corrective retry if invalid
```

The final local schema validation step is required even when the provider claims native schema enforcement.

A provider/model must not be classified as incapable of structured generation solely because it lacks a native strict-schema parameter. If GE can obtain text/JSON and enforce the caller schema locally, the structured operation may still be supportable.

## Corrective retry

A conformance retry is not a transport retry.

```text
transport retry
  retry because the provider operation failed transiently

conformance retry
  retry because the provider operation succeeded but the returned content failed schema validation
```

Corrective feedback must be generic and structural. For example:

```text
The previous output did not satisfy the required schema.
Validation errors:
- $.entities[2].name: required property missing
- $.entities[4].confidence: expected number
Return a corrected JSON value that satisfies the schema.
```

Do not inject product-domain judgments into GenerationEngine repair prompts.

All attempts consume one overall caller-visible deadline/budget. Conformance retries must be bounded.

## Observation direction

The current `InferenceObservation.retry_count` represents provider-operation retry behavior and is not sufficient to describe future structured repair accurately.

The structured-conformance implementation should make the distinction observable. The exact public shape is implementation-owned, but it should preserve truthful concepts equivalent to:

```text
provider_attempt_count
transport_retry_count
conformance_retry_count
```

Usage, cost, and latency should represent the whole structured-generation operation where the provider supplies enough truth to aggregate them.

Do not count a conformance repair as a transport retry.

## Failure semantics

A structured operation is successful only after local schema validation succeeds.

If bounded conformance attempts are exhausted, return normalized `STRUCTURED_OUTPUT_INVALID` (or a deliberately evolved successor failure) with safe diagnostic metadata such as schema name/hash and validation-summary information. Do not retain full prompt/response bodies in the observation by default.

Provider refusal, rate limiting, timeout, and unavailability retain their existing normalized meanings.

## OpenRouter / DeepSeek consequence

The OpenRouter work that motivated this refinement demonstrated why the contract must be above provider-native mechanics.

The Buddy lab has used a DeepSeek model through OpenRouter with JSON-object/instruction/local-validation behavior. A strict native `json_schema` request is therefore not a safe universal definition of OpenRouter structured generation.

Provider adapters must not overclaim structured support based only on OpenAI-compatible request syntax. OpenRouter structured generation in E5B.1 uses JSON instructions, not native `json_schema`.

The bounded lab-direct path remains acceptable when GE cannot yet reproduce required provider-specific routing/reasoning controls without changing experiment semantics.

## Migration consequence for consumers

Once this contract is implemented and settled, consumer migrations should classify existing validation code:

```text
parse/schema/shape validation + model repair
→ candidate to delete in favor of GenerationEngine

domain/business/evidence validation
→ stays with consumer
```

Do not bulk-delete validation merely because it uses Pydantic or JSON. Ownership is determined by what is being validated, not the library used.

## Acceptance tests

The implementation should prove at least:

1. native strict-schema provider path still receives final local validation;
2. non-native JSON/text path can produce a schema-valid result through bounded correction;
3. malformed JSON can be repaired generically;
4. persistent schema mismatch ends as `STRUCTURED_OUTPUT_INVALID`;
5. domain validation remains outside GE;
6. all attempts share one overall deadline;
7. transport and conformance retry counts remain distinguishable;
8. usage/cost/latency accounting is truthful across attempts;
9. tests use domain-neutral schemas only.

## Guiding rule

> **The product owns what the requested structure means. GenerationEngine owns whether the inference output conforms to that requested structure.**
