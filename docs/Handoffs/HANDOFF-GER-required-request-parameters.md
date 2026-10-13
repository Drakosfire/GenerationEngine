# HANDOFF — OpenRouter required request parameters

**Status:** ACTIVE — implementation authorized in this isolated branch.
**Base:** `38d77f33d0e45b3be02459f43add560c2ca2ff8a` (`GenerationEngine` main at dispatch)
**Owner:** GenerationEngine

## Contract

Buddy chooses whether a request requires provider support for every requested
parameter. GenerationEngine owns OpenRouter route translation and provider I/O.
This slice adds `ProviderRoute.require_parameters`, a strict boolean defaulting
to `False`. When true, the OpenRouter adapter forwards `require_parameters: true`
inside its provider-selection object. False preserves the existing wire shape.

This is an endpoint-selection enforcement option. It does not guarantee valid
model output, prove that a particular endpoint supports JSON mode, or explain
the malformed R6 response. OpenRouter may ignore unsupported parameters when
the requirement is not enabled; this change lets a caller request filtering.

## Scope and owned paths

Only these six paths may change:

1. `docs/Handoffs/HANDOFF-GER-required-request-parameters.md`
2. `src/generationengine/types.py`
3. `src/generationengine/providers/openrouter_text.py`
4. `tests/test_openrouter_text.py`
5. `docs/CORE-CONTRACT.md`
6. `docs/CURRENT-STATE.md`

Pin this ACTIVE handoff in the first commit. No other repository, public
capability, provider, prompt, retry, fallback, telemetry, model, output ceiling,
or runtime behavior is in scope.

## Proof obligations

- Reject non-boolean values for `require_parameters` (including integers).
- Prove the actual OpenRouter SDK kwargs include
  `extra_body.provider.require_parameters=true` only when requested.
- Prove omission/default false preserves existing SDK kwargs byte-for-byte at
  the semantic dictionary level, including routing order and fallback behavior.
- Keep provider-neutral route typing and OpenRouter-specific wire mapping at
  their current ownership boundary.
- Run focused tests and the owning shared-type/contract gates; inspect the full
  base-to-head diff before commit and PR.
- Update only the contract and current-state claims affected by this addition.

## Stop conditions

Stop if current main or an active PR owns any scoped file, if implementation
requires another capability or an unlisted path, or if the exact base changes.
Do not run provider calls. Open a PR against `main` and request independent
PRIME review; this prerequisite must be accepted before Buddy consumes it.

## Completion record

Fill in exact commit, PR, review, and disposition after implementation. Until
then this handoff remains ACTIVE.
