# GenerationEngine documentation

GenerationEngine is the **in-process, provider-agnostic inference-execution library** for DungeonMind products.

It owns generic inference execution and inference-call truth. Products own prompt/task meaning, product schemas/domain validation, workflow/orchestration, authorization/quotas, and artifact persistence/publication.

## Active authority

Load the smallest document needed:

- [CORE-CONTRACT.md](CORE-CONTRACT.md) — public semantic contract and ownership boundary.
- [CURRENT-STATE.md](CURRENT-STATE.md) — implemented providers/capabilities/current limitations.
- [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md) — adopted structured-generation semantics.
- [Reports/REPORT-document-authority-audit-2026-09-25.md](Reports/REPORT-document-authority-audit-2026-09-25.md) — documentation cleanup/placement evidence.

Root [README.md](../README.md) is the usage entry point.

## Current settled capabilities

Current `main` includes the accepted E2/E5 contract work through deterministic client cleanup:

- `GenerationClient` public execution facade;
- OpenAI/OpenRouter text and Fal image adapters;
- explicit provider+model targets;
- provider-default temperature via explicit `None`;
- provider-neutral text output-token ceiling;
- schema-less JSON-object ordinary text mode;
- provider-independent structured conformance with bounded repair;
- normalized observations/failures;
- deterministic terminal `await GenerationClient.aclose()`.

There is no active GenerationEngine implementation handoff in this repository at this revision.

## Directory policy

### Active docs

Only current library contract/current-state/adopted decisions belong at `docs/` root.

### `archive/`

Completed E2/E5 handoffs, flag-day compatibility inventories, provider-migration notes, and other historical transition evidence.

Archive content cannot override code/tests/current contract docs.

## Authority rule

For current behavior:

```text
public code/types/tests
→ CORE-CONTRACT / CURRENT-STATE / adopted decision docs
→ merged PR evidence
→ archive / Git history
```

An old handoff that says READY/IN REVIEW is never current authority after its PR settles.

## Ownership rule

GenerationEngine may know:

- provider/model identity and capability;
- generic inference profiles;
- retries/deadlines/provider wire translation;
- structural output conformance;
- normalized failures/usage/cost/latency/provider IDs;
- provider-resource lifecycle.

It must not know product concepts such as statblocks, campaigns, cards, Rules Lawyer, world objects, or Agent turns.

Cross-repository ownership/sequencing authority remains in DungeonOverMind.
