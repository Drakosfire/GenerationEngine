# Structured attempt completion evidence

**Status:** COMPLETE — accepted and merged
**Owner:** GenerationEngine
**Scope:** opt-in evidence hook for non-streaming `GenerationClient.generate_structured()`

## Purpose

Consumers that persist or account for inference attempts need exact evidence for
each provider response GenerationEngine actually receives, including responses
that are refused, malformed, incomplete, or fail schema conformance. A final
operation result and aggregate usage cannot reconstruct those attempts. This
contract adds an awaited, caller-owned sink at the boundary where GE knows a
provider response has returned.

## Contract

`generate_structured(request, *, on_attempt_completed=None)` accepts an optional
async callback. For every returned provider result, GE awaits the callback
strictly before refusal handling, parsing, schema validation, conformance-repair
decisions, or any subsequent provider call. No callback is made for an attempt
that raises without returning a provider result (for example, a transport
failure); operation-level observations remain responsible for truthful attempt
and retry accounting.

Each immutable, provider-neutral event contains:

- an operation identifier stable across the operation's attempts;
- an event/attempt identifier unique within that operation;
- one-based attempt ordinal and kind (`initial` or `conformance`);
- selected provider, requested/resolved model, and response model when known;
- provider request and response identifiers when returned;
- the exact returned text, privately delivered only to the callback;
- input, cached-input, output, and reasoning usage exactly as observed, retaining
  `None` for unknown values rather than substituting zero.
- a response status distinguishing ordinary returned, refused, and incomplete
  provider outcomes.

The callback is opt-in. Event text is not added to `TextResult`, normalized
observations, logs, exceptions, or default telemetry. The contract carries no
prompt, hidden reasoning, domain interpretation, or provider-native schema
semantics.

If the callback raises, GE aborts the operation immediately and performs no
further provider call. GE surfaces a distinct typed attempt-evidence persistence
failure indicating that a provider response was received but its consumer sink
did not accept the event. GE does not retry the callback and does not replay an
uncertain provider attempt. The original callback exception is not exposed as
public error text.

For a provider adapter that reports `PROVIDER_INCOMPLETE` as a typed error,
event metadata/usage/IDs are included but `text` remains `None`: current adapters
do not expose partial response text on that failure path. This is an explicit
source-fidelity limitation, not an assertion that the response body was empty.

Existing structural behavior remains unchanged: at most one conformance
correction is made. Transport retries remain independently governed by the
existing request/deadline policy; this callback is not an authorization ceiling.
Every actual returned response is accounted for. No live provider invocation was used in this contract work.

## Acceptance record

- PR: [#19](https://github.com/Drakosfire/GenerationEngine/pull/19)
- Reviewed head: `a4bd256b5ffd9dfbc3d41ea6f88e83fac242a428`
- Review: ARCH APPROVE; PRIME review comment `6158161330`
- Merge commit: `e4b683a459d9f0dfaa2a41d8b073ff6117c3aa6e`

## Minimal write set

- `docs/Handoffs/HANDOFF-structured-attempt-completion-evidence.md` (this
  pinned consumer contract; first commit)
- `src/generationengine/client.py`
- `src/generationengine/types.py`
- `src/generationengine/failures.py`
- `src/generationengine/__init__.py` only for the public event/error types
- `tests/test_structured_conformance.py`
- `docs/CURRENT-STATE.md` only to record the accepted active capability

No provider adapter, provider/native schema behavior, Buddy consumer, prompt,
serving runtime, or product/domain contract changes are in scope.

## Acceptance proof

Fake-provider tests must prove callback-before-parse/refusal/repair ordering,
complete two-response/one-correction accounting, final-invalid response
accounting, sink-failure stop-before-follow-up, no response event for transport
failures without a returned body, refusal/incomplete response inclusion when a
result is returned, stable IDs and attempt kinds/ordinals, and exact preservation
of null usage. No live provider is used.

This handoff is complete. Further consumer integration is a separate capability and is not authorized by this handoff.
