# E5H — provider-neutral text output token ceiling

**Status:** MERGED — GenerationEngine PR #8 at `f502c9883013d3ec9b866ca9276dfd7def141599`
**Architecture owner:** `Drakosfire/DungeonOverMind`
**Execution repository:** `Drakosfire/GenerationEngine`
**Canonical design:** DungeonOverMind `Docs/Plans/HANDOFF-E5H-generationengine-text-output-token-ceiling.md` at `17697a6723165f29f264d7cc203c9d2870bcd687`
**Implementation base:** `80288d7b467ac3c3586f4e3c964385cefe69f931`
**Predecessor:** Buddy E5G PR #746 merged at `9510a6dbdfde52917e249710724a35c089410f68`
**PR topology:** `serial`
**Assigned branch:** `codex/e5h-text-output-token-ceiling`
**Assigned PR title:** `E5H: add provider-neutral text output token ceiling`
**Final reviewed head:** `bdcaad5c400c32d9f813decc88cfa2fd3840fd78`
**Review cycles:** 2
**Accepted evidence:** focused contract/provider 94 passed; full provider-free 125 passed; Python 3.11/3.13 CI green; build/wheel import/lock/sync/Ruff/diff-check green

## Mission

Add one optional provider-neutral output ceiling to GenerationEngine text
execution:

```python
max_output_tokens: int | None = None
```

The exact field is present on both `TextRequest` and `TextGenerationCall`.
Omitted or `None` means GE sends no provider output-token ceiling. A positive
integer requests that exact ceiling. Zero and negative values are invalid.

This contract applies unchanged to ordinary text, structured text, structured
repair, transport retry, and streaming. No consumer repository moves in E5H.

## Re-anchor and topology

Activation re-anchor on 2026-09-23 established:

- GenerationEngine `origin/main` is exactly `80288d7b...`;
- there are no open GenerationEngine PRs;
- the accepted public/core contract still lacks an output ceiling;
- OpenAI uses Responses and OpenRouter uses Chat Completions;
- no competing generic output-limit field exists.

Topology is `serial`. Open only the assigned E5H PR. No Buddy migration,
dependency-pin update, or successor PR is authorized.

## Write lease

Expected modified paths:

| Action | Path | Purpose |
|---|---|---|
| Modify | `src/generationengine/types.py` | public request field and validation |
| Modify | `src/generationengine/providers/base.py` | provider-neutral call field |
| Modify | `src/generationengine/client.py` | propagate the exact value through every text call |
| Modify | `src/generationengine/providers/openai_text.py` | Responses wire mapping |
| Modify | `src/generationengine/providers/openrouter_text.py` | Chat Completions wire mapping |
| Modify | `tests/test_client.py` | validation, propagation, retry, and repair witnesses |
| Modify | `tests/test_openai_text.py` | omit/forward/stream/structured wire witnesses |
| Modify | `tests/test_openrouter_text.py` | omit/forward/stream/structured wire witnesses |
| Modify | `docs/CORE-CONTRACT.md` | public semantic contract |
| Modify if useful | `docs/CURRENT-STATE.md` | current capability inventory |

Do not modify dependency/version files, observation, catalog, resolver,
failures, retry/deadline policy, pricing, profiles, provider identity, or any
consumer repository.

## Implementation contract

Public/core fields:

```python
max_output_tokens: int | None = Field(default=None, ge=1)
```

`GenerationClient._text_call` copies the exact request value. Structured repair
builds its replacement prompt through the same call constructor and therefore
inherits the exact ceiling. Transport retry reuses the same call. No numeric
default, catalog/profile population, provider branching in the client, or local
truncation is allowed.

Provider adapters translate only at the wire:

```text
OpenAI Responses:   max_output_tokens=N
OpenRouter Chat:    max_completion_tokens=N
```

Both adapters omit their wire field entirely for `None`. Streaming uses the
same provider request builder and carries the exact value to the provider.

Do not add the requested ceiling to `InferenceObservation`; actual
`output_tokens` remains measured usage truth.

## Merge-blocking witnesses

- Both public/core fields are `int | None`, default `None`, minimum 1.
- Zero and negative requests fail validation before provider execution.
- Omitted callers retain their existing wire shapes.
- Exact positive values reach ordinary and structured provider calls.
- Structured corrective attempts inherit the original exact ceiling.
- Transport retries inherit the original exact ceiling.
- OpenAI omits `None` and forwards `N` as `max_output_tokens=N` for ordinary,
  structured, and streaming calls.
- OpenRouter omits `None` and forwards `N` as `max_completion_tokens=N` for
  ordinary, structured, and streaming calls.
- Observation, catalog, resolver, profile, provider identity, retry, deadline,
  pricing, and failure contracts remain unchanged.
- No consumer repository changes.
- `docs/CORE-CONTRACT.md` documents the generic field and omission semantics.

## Verification

Run on the final exact head:

```bash
uv lock --check
uv sync
uv run python -c "import generationengine; print(generationengine.__version__)"
uv run ruff check src/generationengine/types.py src/generationengine/providers/base.py src/generationengine/client.py src/generationengine/providers/openai_text.py src/generationengine/providers/openrouter_text.py tests/test_client.py tests/test_openai_text.py tests/test_openrouter_text.py
uv run pytest -q tests/test_client.py tests/test_openai_text.py tests/test_openrouter_text.py tests/test_structured_conformance.py tests/test_explicit_targets.py
uv run pytest -q
git diff --check
git diff --name-only 80288d7b467ac3c3586f4e3c964385cefe69f931...HEAD
```

No live provider call is required.

## State sync after merge

After merge, record the PR, exact reviewed head, merge SHA, review-cycle count,
and accepted evidence in this handoff and the DungeonOverMind canonical E5H
handoff/roadmap. Only then design the separate Buddy grounded-answer migration
and its deliberate GE pin update.

## Stop conditions

Stop rather than widen scope if current provider contracts changed materially;
a concurrent PR overlaps the lease; the pinned SDK cannot express OpenRouter's
verified field; provider/model-specific branching is required; streaming cannot
carry the ceiling; structured repair cannot inherit it; or a consumer must
change for proof.
