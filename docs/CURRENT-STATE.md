# GenerationEngine current state

**Status:** current `main` implementation reference  
**Contract:** [CORE-CONTRACT.md](CORE-CONTRACT.md)  
**Structured conformance:** [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md)

## Public execution surface

```text
GenerationClient
  generate_text
  generate_structured
  decide
  stream_text
  generate_image
  edit_image
  aclose
```

Public request/result/failure/observation types are exported from `generationengine`.

## Typed decisions

The generic `DECISION` capability has provider-neutral Binary, Choice, and Score questions/results and a distinct `DecisionProvider` execution seam. `decide()` requires an explicit registered decision provider and model, validates JSON-safe state and typed answer alignment, and applies the existing overall deadline, transport retry, and normalized failure rules. The optional `typesafe` extra wires the first provider adapter through the official async SDK and the Vercel AI Gateway route with SDK retries disabled. The adapter reads `TYPESAFE_JEV_API_KEY` itself. No text fallback or decision profile is added.

### Prepared dispatch gate

The optional TypeSafe adapter now accepts a product-owned `DecisionDispatchGate`
from `generationengine.decision_dispatch`. It authorizes actual SDK-serialized
body bytes before HTTP I/O and records sending, known-not-sent, response-header
acknowledgment and uncertain outcomes. This mode disables compression, redirects
and all automatic retries; failed lifecycle acknowledgment fails execution.
The TypeSafe extra requires SDK 0.7.5 for supported compression control.
Actual SDK serialization, concurrent calls and failure boundaries are tested with
mock HTTP transports; this feature does not claim a live provider witness or
product adoption. See the [dispatch contract](CORE-CONTRACT.md#opt-in-decision-dispatch-authorization).

## Text execution

Registered text providers:

- `openai` → OpenAI Responses adapter;
- `openrouter` → OpenRouter Chat Completions adapter.

Selection supports:

- governed profile/catalog resolution;
- explicit `provider + model` execution without requiring a catalog row.

`TextRequest.temperature` semantics:

```text
omitted          GE default 0.7
explicit number  forward exact value, including 0.0
explicit None    omit provider field; provider/model default owns sampling
```

`TextRequest.max_output_tokens`:

```text
None / omitted   no provider ceiling
positive int     exact provider-neutral output-token ceiling
<= 0             invalid before provider execution
```

Provider adapters translate the generic ceiling into provider-native wire vocabulary. GE does not locally truncate text.

`TextRequest.reasoning_effort=None` omits provider reasoning controls; an
explicit label reaches OpenAI Responses or OpenRouter Chat unchanged, including
structured repair and streaming. On OpenRouter, `"none"` also emits
`reasoning.enabled=false`. An optional typed `provider_route` constrains
OpenRouter's upstream provider order and fallback behavior; it composes with
reasoning controls and survives structured repair and transport retries. Its
strict `require_parameters` option defaults false; true asks OpenRouter to
filter endpoints lacking any requested parameter, and is sent only when true.
This filters route eligibility but does not guarantee valid generated output.
A route for another selected GE text provider fails before its adapter is
constructed. Omitting the route keeps the existing provider request shape.

`max_transport_retries=None` preserves two transport retries after the first
attempt. Explicit nonnegative values set the
ceiling per non-streaming provider execution within the overall deadline.
Streaming accepts `None` or `0`; a positive value fails before provider work.

### Schema-less JSON-object mode

Ordinary `generate_text()` supports `json_object=True`.

```text
provider asked for JSON object
→ raw provider text returned
→ TextResult.parsed remains None
→ no local JSON parse/schema validation/repair
```

Products own parsing/domain validation/fallback for this mode.

`json_object=True` is incompatible with `json_schema`, and JSON-object streaming is rejected before provider execution.

### Prepared ordinary-text dispatch gate

The OpenAI adapter accepts opt-in `TextDispatchGate` from
`generationengine.text_dispatch`. It authorizes exact SDK-serialized nonstreaming
Responses bytes with an explicit empty tool list and no automatic transport
retries, redirects or ambient proxy routing. Actual headers/outcome uncertainty
are distinct from response-body parsing. Streaming and structural-repair requests
are rejected on a guarded provider. Existing unguarded execution is unchanged.
Synthetic real-SDK/mock-HTTP tests prove the boundary; live provider and product
adoption remain separate. See [text dispatch contract](CORE-CONTRACT.md#opt-in-text-dispatch-authorization).

## Structured generation

`generate_structured()` implements provider-independent structural conformance:

```text
caller JSON Schema
→ provider-specific generation strategy
→ GE local JSON parse
→ GE local JSON Schema validation
→ at most one structural corrective inference retry
→ parsed schema-conforming object OR normalized failure
```

Callers may opt into an awaited `on_attempt_completed` evidence callback for
each returned provider response. The hook runs before refusal handling,
conformance checks, or a corrective follow-up; it is not called for transport
attempts that return no provider result. See the accepted
[structured attempt evidence handoff](Handoffs/HANDOFF-structured-attempt-completion-evidence.md).

Provider-native strict-schema support is an optimization, not the semantic contract.

OpenAI Responses `status == "incomplete"` on non-streaming generation is a
terminal `PROVIDER_INCOMPLETE` failure before parse or repair. Refusal retains
precedence. Safe provider IDs and reported usage, including reasoning tokens,
remain in the failure observation; partial response text does not. Streaming
semantics are unchanged.

Products retain domain/business/evidence validation.

## Streaming

`stream_text()` is transport-neutral and yields deltas followed by exactly one terminal event. Streaming does not perform transport retry. Unsupported/invalid requests return a failed terminal rather than provider execution.

## Images

Fal-backed image generation/editing returns bytes through `ImageResult`.

GenerationEngine does not publish artifacts, upload to Cloudflare, or return product URLs.

## Observations and failures

`InferenceObservation` records inference-call truth across success/failure:

- provider/requested/resolved/response model IDs;
- optional intermediary `provider_transport`, without inferring an upstream concrete model;
- provider request/response IDs when known;
- input/cached/output token usage when known;
- provider-reported reasoning tokens when known (`None` remains unknown);
- cost when known;
- operation latency;
- transport retry count;
- structured conformance retry count;
- provider attempt count;
- normalized state/failure code.

Unknown values remain `None`; they are not invented as zero.

Public failures use `GenerationEngineError` + normalized `InferenceFailure` / `FailureCode`, never provider SDK exceptions.

## Client lifecycle

`await GenerationClient.aclose()` is terminal and idempotent.

It:

- closes instantiated closeable providers once;
- does not construct lazy providers just to close them;
- attempts all instantiated providers even if one close fails, then re-raises the first cleanup error;
- causes later inference to fail with `INVALID_REQUEST` before provider execution.

Built-in OpenAI/OpenRouter adapters close their underlying async SDK clients.

## Current limitations / intentional boundaries

- no product-specific generation methods;
- no Agent runtime;
- no network GenerationEngine service;
- no artifact publication/storage;
- no embeddings/transcription/speech/moderation contract yet;
- no JSON-object streaming;
- provider-specific experimental controls not represented by the generic contract may require a bounded lab-only direct path.

Deleted E2-era facades and cutover details are historical under `docs/archive/`.
