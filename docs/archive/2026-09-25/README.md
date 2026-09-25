# GenerationEngine transition archive — 2026-09-25

**Status:** HISTORICAL TRANSITION EVIDENCE  
**Current authority:** `docs/README.md`

This archive removes settled migration/cutover documents from the active contract surface.

## E2

Archived:

- flag-day compatibility/deleted-API inventory;
- E2 successor-slice sequencing.

GitHub truth:

| PR | Outcome | Merge commit |
|---:|---|---|
| 1 | E2A contract characterization | `cbb03c14f7d4ec9dc1b27e18e80e5f5bb3049295` |
| 2 | E2B trustworthy core primitives | `0414723ce91625df08ba7059842d0f10722f19b4` |
| 3 | coordinated GenerationClient cutover | `0d01547e2d9afec68e87b4c8f7e6aaa047e8c42a` |

Deleted E2-era facades remain deleted. Current public API truth is in code, `CORE-CONTRACT.md`, and `CURRENT-STATE.md`.

## E5 reusable contract refinements

Archived completed execution handoffs:

| PR | Capability | Merge commit |
|---:|---|---|
| 4 | explicit targets + OpenRouter dispatch | `eb388980b9c75b92e67d046afb50a0d1418fd28c` |
| 5 | structured conformance | `eba1eb7f7c62ed2f3dd4f77f42a8cfab0a8d4773` |
| 6 | conformance acceptance witnesses | `63196b7f31528ff57afc6afac8723e6faf0f820e` |
| 7 | provider-default temperature | `80288d7b467ac3c3586f4e3c964385cefe69f931` |
| 8 | provider-neutral output-token ceiling | `f502c9883013d3ec9b866ca9276dfd7def141599` |
| 9 | schema-less JSON-object mode | `7f5c94f7fc24051cd62595f836b2db4105d68d50` |
| 10 | deterministic GenerationClient cleanup | `9122257f5a8842e4771990a3316130bc1bf7e332` |

The E5H and E5J handoff bodies are preserved here as implementation history. Their semantics are now part of the active core contract/current state, not active handoffs.

## Restore rule

Do not move these files back to the active docs root merely for discoverability. If a behavior changes, update the active contract/current-state docs and tests; use this archive only for transition history.
