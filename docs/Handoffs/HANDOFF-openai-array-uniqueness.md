# OpenAI native array uniqueness projection

Status: ACTIVE serial implementation; PRIME owns this bounded provider fix.
Base: GenerationEngine `2ef4a6d463b01cac193ce59e3959ada7a82d7d73`.
Branch: `codex/openai-array-uniqueness`; reuse the existing GE checkout.

ADVERSARY's ordinary c1-s01/Luna cell rejected its first native Responses schema
with HTTP400 `invalid_json_schema`: nested `source_quote_ids.uniqueItems` is not
permitted. Preserve that original failure under
`model-cell-c1-s01-luna-ee9f0bb898/provider-evidence/worker-2245951/`.
This is provider wire compatibility, not semantic extraction quality.

Contract: OpenAI receives a schema projection without unsupported array uniqueness
keywords. GE retains the exact original caller schema for local validation and
bounded structural repair. No product schema/prompt/gold/pass ordering, target
selection, fallback or retry budget changes. No live provider calls in this slice.
Schema traversal must preserve property names and literal data.

Exclusive write set: `src/generationengine/utils/schema_utils.py`,
`tests/test_openai_array_uniqueness.py`, `docs/CURRENT-STATE.md`, this handoff.
No ports, databases, consumer pins or benchmark fixtures are modified here.

Verify real SDK/mock HTTP serialized schema, original immutability, nested schema
positions and same-named properties/literals; reject duplicate output locally
with repair disabled and exercise existing bounded repair. Run provider/core
regressions, inspect cumulative diff, independent Bugbot, hosted gates, PRIME
merge. Return exact reviewed and merged pins to ADVERSARY; Buddy dependency
adoption is a separate bounded consuming slice. Lease ends on merge; settle
post-merge facts in PR handback and the next dependent handoff.

References: official OpenAI Structured Outputs and Prompt generation guides,
which document the subset and unsupported `uniqueItems` keyword; existing
`docs/STRUCTURED-CONFORMANCE.md` owns local conformance.
