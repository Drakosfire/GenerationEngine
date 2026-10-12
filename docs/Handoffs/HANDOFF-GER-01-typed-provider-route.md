# HANDOFF — GER-01 typed provider route

**Status:** ACTIVE — implementation authorized
**Repository:** `Drakosfire/GenerationEngine`
**Base:** remote `main` `128710668369b73beff78ea0e4377ec252cd04e4`
**PR topology:** serial; one GenerationEngine implementation PR
**Owner:** GenerationEngine text execution and OpenRouter adapter
**Consumer:** a later, separately authorized Buddy experiment migration

## Capability

One text request can explicitly constrain an upstream provider route through a typed, reusable `ProviderRoute`. The route survives ordinary text, JSON-object text, structured repair, and transport retry. OpenRouter translates it to its provider-routing wire object. Unroutable providers reject the request before an SDK call. Requests without a route keep their current behavior.

`ProviderRoute` has a nonempty ordered tuple of distinct, nonblank provider names and a required boolean `allow_fallbacks`. No model, product, or vendor name is embedded in GenerationEngine policy. The existing `reasoning_effort="none"` means reasoning disabled; OpenRouter emits both `effort="none"` and `enabled=false` to preserve the explicit disabled wire form.

GenerationEngine retains its existing retry, timeout, conformance, failure and observation rules. Provider routing adds no arbitrary `extra_body` passthrough and no catalog/default selection change. No Buddy migration, pricing/cost observation feature, live provider call, or credential change is part of this slice.

## Exact write lease

Production:

- `src/generationengine/types.py`
- `src/generationengine/providers/base.py`
- `src/generationengine/client.py`
- `src/generationengine/providers/openrouter_text.py`

Tests:

- `tests/test_client.py`
- `tests/test_openrouter_text.py`
- `tests/test_structured_conformance.py`

Authority:

- `docs/CORE-CONTRACT.md`
- `docs/CURRENT-STATE.md`
- `docs/Handoffs/HANDOFF-GER-01-typed-provider-route.md`

No other path is leased. The public type may be imported from `generationengine.types`; the existing package export file is outside this lease.

## Acceptance witnesses

1. Typed route validation rejects empty, blank, duplicate, and nonboolean fallback values. Both `TextRequest` and `TextGenerationCall` preserve a valid tuple.
2. A fake OpenRouter SDK sees the exact model, JSON-object response format, no temperature, `provider.order`, `allow_fallbacks=false`, and reasoning `effort=none, enabled=false` on one call.
3. An absent route leaves the OpenRouter request unchanged except for the separately explicit `reasoning_effort="none"` mapping. An explicit route for OpenAI or another unsupported provider fails before provider SDK execution; invalid route values fail construction.
4. Structured repair and transport retries forward the same route. Their observation counts remain the number of actual provider calls, including terminal failure.
5. Focused owning tests and repository CI gates pass; cumulative base-to-head diff remains within the lease.

## Handoff boundary

The later Buddy consumer may use an explicit OpenRouter model and route while retaining its product prompts, parse rules, attempt budget, receipts, and cost interpretation. Current GE observations do not report OpenRouter provider-supplied `usage.cost`; the consumer must not claim that value without a separate accepted contract.
