# GenerationEngine current state

**Status:** current `main` implementation reference  
**Contract:** [CORE-CONTRACT.md](CORE-CONTRACT.md)  
**Structured conformance:** [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md)

## Public execution surface

```text
GenerationClient
  generate_text
  generate_structured
  stream_text
  generate_image
  edit_image
  aclose
```

Public request/result/failure/observation types are exported from `generationengine`.

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
structured repair and streaming. `max_transport_retries=None` preserves two
transport retries after the first attempt. Explicit nonnegative values set the
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

Provider-native strict-schema support is an optimization, not the semantic contract.

Products retain domain/business/evidence validation.

## Streaming

`stream_text()` is transport-neutral and yields deltas followed by exactly one terminal event. Streaming does not perform transport retry. Unsupported/invalid requests return a failed terminal rather than provider execution.

## Images

Fal-backed image generation/editing returns bytes through `ImageResult`.

GenerationEngine does not publish artifacts, upload to Cloudflare, or return product URLs.

## Observations and failures

`InferenceObservation` records inference-call truth across success/failure:

- provider/requested/resolved/response model IDs;
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
