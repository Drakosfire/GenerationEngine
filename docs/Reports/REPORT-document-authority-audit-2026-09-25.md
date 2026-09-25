# Report — GenerationEngine Documentation Authority Audit, 2026-09-25

**Status:** COMPLETE — first rigor pass  
**Repository:** `Drakosfire/GenerationEngine`

## Result

GenerationEngine now has a deliberately small active documentation surface matching its actual role:

> **provider-agnostic, in-process inference execution and inference-call truth**

Products retain prompt/task meaning, product schemas/domain validation, orchestration, authorization/quotas, and artifact persistence/publication.

## Current active documentation shape

At closeout:

```text
11 files under docs/
6 historical files under docs/archive/
5 active documentation files
```

Active:

- `docs/README.md` — documentation authority/index;
- `docs/CORE-CONTRACT.md` — public semantic contract;
- `docs/CURRENT-STATE.md` — implemented capabilities/current limitations;
- `docs/STRUCTURED-CONFORMANCE.md` — active adopted structured-generation decision;
- this audit report.

## Historical transition material archived

Moved out of the active docs root:

- E2 successor sequencing;
- E2 flag-day compatibility/deleted-API inventory;
- E5H output-token-ceiling implementation handoff;
- E5J JSON-object-mode implementation handoff.

Those files remain under `docs/archive/2026-09-25/transition/` with an archive ledger recording exact PR/merge evidence.

Historical OpenAI Responses migration notes imported from DungeonOverMind are consolidated under `docs/archive/2026-09-25/historical-imports/` and are explicitly not current contract authority.

## PR truth reconciled

No GenerationEngine PR is open at this audit.

Verified merged sequence:

| PR | Capability | Merge commit |
|---:|---|---|
| 1 | E2A contract characterization | `cbb03c14f7d4ec9dc1b27e18e80e5f5bb3049295` |
| 2 | E2B core primitives | `0414723ce91625df08ba7059842d0f10722f19b4` |
| 3 | GenerationClient cutover | `0d01547e2d9afec68e87b4c8f7e6aaa047e8c42a` |
| 4 | explicit targets / OpenRouter | `eb388980b9c75b92e67d046afb50a0d1418fd28c` |
| 5 | structured conformance | `eba1eb7f7c62ed2f3dd4f77f42a8cfab0a8d4773` |
| 6 | conformance acceptance witnesses | `63196b7f31528ff57afc6afac8723e6faf0f820e` |
| 7 | provider-default temperature | `80288d7b467ac3c3586f4e3c964385cefe69f931` |
| 8 | output-token ceiling | `f502c9883013d3ec9b866ca9276dfd7def141599` |
| 9 | JSON-object mode | `7f5c94f7fc24051cd62595f836b2db4105d68d50` |
| 10 | deterministic client cleanup | `9122257f5a8842e4771990a3316130bc1bf7e332` |

## Current semantics refreshed

`CURRENT-STATE.md` now records current main behavior rather than the old structured-conformance feature branch:

- OpenAI/OpenRouter text dispatch;
- explicit provider+model lane;
- explicit `temperature=None` semantics;
- `max_output_tokens` semantics;
- schema-less `json_object=True` ordinary text behavior;
- structured conformance + one bounded repair;
- streaming boundary;
- byte-returning Fal image behavior;
- observation/failure truth;
- terminal/idempotent `GenerationClient.aclose()`.

`CORE-CONTRACT.md` is now labeled as implemented rather than a target/future contract.

## Source-level documentation corrected

Developer-facing module docstrings were also stale and were corrected:

- `failures.py` no longer describes a future E2B target taxonomy;
- `providers/base.py` now describes live provider-neutral seams;
- `observation.py` no longer labels the live type as an E2A/E2B transitional artifact.

## Executable authority

`tests/test_core_contract_invariants.py` now asserts:

- the exact active docs-root authority set exists and is non-empty;
- extra active docs cannot silently accumulate outside that set;
- current contract docs do not resurrect known stale future/feature-branch language.

The existing repository CI already runs the full pytest suite on Python 3.11 and 3.13, so documentation authority is enforced through the normal test matrix rather than a separate documentation-only workflow.

## Placement rule going forward

```text
generic inference execution contract/current state
→ GenerationEngine active docs

provider/model execution implementation
→ GenerationEngine code/tests

product prompts/schemas/domain/workflows
→ owning product repository

cross-repository ownership/sequencing
→ DungeonOverMind

completed migration handoffs
→ docs/archive or Git/PR history
```

Historical transition volume is acceptable. Competing active authority is not.


## Final contract-drift corrections

The final rigor sweep also corrected two active-document mismatches:

- structured-conformance observation/retry language now describes the implemented counters rather than a future direction;
- the image contract now matches current public types: `GeneratedImage` contains bytes/media/size while `ImageResult` owns the `InferenceObservation`.

Streaming/core docs were also normalized from E2 migration narration to present-tense public semantics.
