# GenerationEngine core contract

**Status:** IMPLEMENTED — live execution is `GenerationClient` over OpenAI, OpenRouter, and Fal adapters.  
**Current behavior:** [CURRENT-STATE.md](CURRENT-STATE.md)

GenerationEngine is an in-process inference capability. It is not a product backend, not a network service, and not an Agent runtime.

---

## Ownership

Products own:

```text
product prompts
product schemas and domain validation
product workflow
product authorization / quotas
product persistence and artifact publication policy
product-facing error translation
Agent loops and context assembly
mapping of product actions to generic inference profiles
```

GenerationEngine owns:

```text
provider adapters and provider endpoints
provider/model capability catalog
generic inference profiles
provider/model resolution
text generation
structured generation
transport-neutral text streaming
image generation
image editing / inpainting
retry / timeout behavior
provider error normalization
typed probabilistic decision execution (provider-neutral Binary, Choice, Score)
usage normalization
pricing and inference-call cost
provider request IDs
inference latency
safe inference observations
```

GenerationEngine must not know what a statblock, card, character, store, campaign, Runbook, or Agent turn is.

---

## 1. Capability surface

Core capabilities, independent of product concepts:

```text
text
structured text
streaming text
image generation
image editing / inpainting
typed decision
```

Do not add methods named `generate_statblock`, `generate_card`, `generate_map`, or `generate_character`.

Embeddings, transcription, speech, and moderation are out of this contract until an explicit ownership decision adds them.

### Typed decision capability

`GenerationClient.decide(DecisionRequest)` is separate from text and structured-text generation. A request supplies JSON-safe state, one or more uniquely named Binary, Choice, or Score questions, an explicit provider and model, and optional overall deadline/transport-retry controls. Choice options and Score levels are ordered, distinct labels; Score requires at least two levels. Choice questions may include a description for every option to preserve the caller's meaning. Products own the question wording and label meaning.

The `DecisionProvider` protocol receives a provider-neutral `DecisionGenerationCall` and returns typed answers plus provider-call metadata. The client validates exact answer-name/kind alignment, choice membership, distribution labels/probabilities, and finite scores. A malformed provider answer becomes `MALFORMED_PROVIDER_RESPONSE`; invalid caller input becomes `INVALID_REQUEST`. A provider capability not wired for decisions becomes `UNSUPPORTED_CAPABILITY`. There is no text-generation fallback or generic decision profile in this slice. One `decide()` operation yields one `InferenceObservation` across transport retries.

Optional answer confidence, binary true probability, Choice probabilities, and Score distribution preserve what a provider reports. Their absence remains `None`; GenerationEngine does not invent calibration. Provider adapters may use provider-native controls, but those controls and SDK types remain behind the adapter seam.

The first adapter is the optional `typesafe` extra. It maps these public forms to the official SDK's three question families inside the provider module, uses the async SDK with its internal retries disabled, and routes through the Vercel AI Gateway TypeSafe endpoint. It reads `TYPESAFE_JEV_API_KEY` in GenerationEngine. A scalar JSON state is valid in the generic contract but is rejected as `INVALID_REQUEST` by this adapter because the current TypeSafe API accepts only text, object, or array state. `provider_transport=vercel_ai_gateway` identifies the route and does not imply a concrete upstream model version.

Hosting remains in-process. A network GenerationEngine service is not part of the current contract.

### Text temperature

`TextRequest.temperature` and `TextGenerationCall.temperature` are `float | None` and default to `0.7`. The three states are distinct:

```text
omitted field     existing GenerationEngine default 0.7; provider receives 0.7
explicit number   forward that exact numeric value, including 0.0
explicit None     omit the provider temperature field; provider/model default owns sampling
```

This applies to ordinary text, structured text, structured repair, and streaming for OpenAI and OpenRouter. A structured corrective attempt inherits the original request temperature unchanged. GenerationEngine does not invent a provider-specific default temperature.

### Text output-token ceiling

`TextRequest.max_output_tokens` and `TextGenerationCall.max_output_tokens` are
provider-neutral `int | None` fields with a minimum value of 1 and a default of
`None`:

```text
omitted / None      send no provider output-token ceiling
positive integer N  request exactly N maximum generated output tokens
zero / negative     invalid request
```

The exact caller value applies to ordinary text, structured text, structured
repair attempts, transport retries, and streaming. GenerationEngine does not
invent a numeric default, expand the ceiling for repair, or truncate output
locally. Provider adapters own wire vocabulary: OpenAI Responses receives
`max_output_tokens`, while OpenRouter Chat Completions receives
`max_completion_tokens`.

The requested ceiling is configuration, not measured usage, and is therefore
not part of `InferenceObservation`. Actual provider-reported `output_tokens`
remains observation truth.

### Reasoning effort and transport retries

`TextRequest.reasoning_effort` and `TextGenerationCall.reasoning_effort` are
`str | None` and default to `None`. The default omits provider reasoning
controls. An explicit effort label is forwarded unchanged for ordinary,
structured, repair, and streaming text. OpenAI Responses receives
`reasoning={"effort": label}`; OpenRouter Chat receives the same reasoning
object through adapter-local `extra_body`. For OpenRouter, explicit `"none"`
also sends `"enabled": false`. Model-specific support is decided by the
provider.

`TextRequest.max_transport_retries` and `TextGenerationCall.max_transport_retries`
are nonnegative `int | None` fields. `None` preserves the existing ceiling of
two transport retries after the first attempt. An explicit `N` permits at most
`N` transport retries for each non-streaming provider execution, within the
same overall deadline. Structured repair uses the same ceiling for its own
provider execution. Retryability and backoff are unchanged; provider SDK
retries remain disabled.

Streaming remains no-retry. `None` and `0` preserve streaming behavior. A
positive retry ceiling yields one `INVALID_REQUEST` terminal before provider
construction, with zero provider attempts.

### Explicit provider route

`TextRequest.provider_route` and `TextGenerationCall.provider_route` are
optional typed `ProviderRoute` values. A route has a nonempty ordered tuple of
distinct, nonblank upstream provider names and an explicit boolean
`allow_fallbacks`, plus a strict `require_parameters` boolean defaulting to
`False`. Names with outer whitespace, duplicate names ignoring case, and
unknown route fields are invalid. An absent route leaves provider routing
unspecified and preserves existing request behavior.

OpenRouter Chat maps the route to its provider-selection object inside
adapter-local `extra_body`. It composes with reasoning controls, including
`reasoning_effort="none"`. When `require_parameters=True`, OpenRouter receives
that option and filters route endpoints that do not support every requested
parameter. The default false value is omitted from the wire and preserves
existing request shapes. This constrains endpoint selection; it does not
guarantee generated content or prove that any endpoint supports the request. A
route whose selected GE text provider is not OpenRouter fails as
`INVALID_REQUEST` before provider construction or execution, with zero provider
attempts; streaming emits one matching failure terminal.
The route applies unchanged to ordinary, JSON-object, structured, repair,
transport-retry, and streaming calls. It does not select a GE model, add a
catalog record, or expose arbitrary provider request fields.

### Schema-less JSON-object mode

`TextRequest.json_object` and `TextGenerationCall.json_object` are
provider-neutral booleans that default to `False`. When true on ordinary
`generate_text()`, GE asks the provider for a valid JSON object but returns the
provider's raw text with `TextResult.parsed` remaining `None`. Products retain
parsing, domain validation, and fallback ownership; GE performs no local parse
or conformance repair for this mode.

Provider adapters map the generic request to their native wire controls:

```text
OpenAI Responses  text.format = {"type": "json_object"}
OpenRouter Chat   response_format = {"type": "json_object"}
```

False omits those controls and preserves existing request shapes.
`json_object=True` is mutually exclusive with `json_schema`, and streaming
JSON-object mode is not supported; both combinations fail with
`INVALID_REQUEST` before provider execution. `generate_structured()` remains
the JSON Schema validation and conformance-repair API.

### Client lifecycle

`await GenerationClient.aclose()` deterministically releases resources owned by
providers already instantiated on that client. Cleanup is terminal and
idempotent: unique closeable provider objects are closed once, lazy providers
are not constructed merely for cleanup, and later inference fails with
`INVALID_REQUEST` before provider execution.

Cleanup attempts every instantiated provider even if an earlier close fails,
then re-raises the first cleanup exception unchanged. Cleanup errors are not
inference failures and do not create observations. Built-in OpenAI and
OpenRouter text adapters delegate cleanup to their underlying async SDK
clients.

---

## 2. Provider boundary

Capability-focused protocols, not one giant provider type:

```text
TextProvider
ImageProvider
DecisionProvider
```

After the provider reset:

- the core/service layer must not instantiate OpenAI, OpenRouter, or Fal SDK clients
- provider SDK exception types must not be the public contract
- advertised providers must match registered wiring and declared extras
- provider identity is the dispatch/observation name (`openai`, `openrouter`, `fal`); it is not the SDK used internally

OpenRouter remains `provider=openrouter` even when the adapter reuses an OpenAI-compatible Python SDK or request shape.

`TextProvider` and `ImageProvider` are the live provider protocols. OpenAI/OpenRouter text and Fal image execution run behind these seams. Text dispatch uses provider identity.

---

## 3. Model selection boundary

```text
product action                 # owned by the consumer
        ↓
generic inference profile      # GenerationEngine vocabulary
        ↓
provider + model resolution    # GenerationEngine-owned
```

There are two intentional execution lanes:

```text
governed resolution            explicit target
profile / catalog              provider + model
GE chooses a known target      caller chooses target
catalog is required            catalog entry is not
```

<!-- ACCEPTED_PROFILES -->
```text
text_fast
structured_low_cost
structured_high_reliability
image_high_quality
image_edit_high_quality
```
<!-- /ACCEPTED_PROFILES -->

These names describe inference requirements. Additional generic profiles may be added later if they remain requirement-shaped (`image_fast`, `text_high_reliability`). They must not encode product actions.

Forbidden as GenerationEngine profile names:

```text
statblock_generation
map_prompt_compilation
ruleslawyer_response
agent_turn
card_generation
character_generation
```

Resolution precedence:

```text
1. explicit provider + explicit model
   → caller-selected target; no model-catalog membership required
2. explicit model without provider
   → strict catalog model resolution
3. profile without explicit target
   → catalog/profile resolution
4. provider without model
   → INVALID_REQUEST
5. neither profile nor model
   → INVALID_REQUEST
```

If GenerationEngine chooses the model, the model must be cataloged. If the caller explicitly chooses both provider and model, the provider must be registered and the model catalog entry is not required.

Do not silently interpret an unknown model string as an OpenAI model merely because the OpenAI SDK is installed. A profile, when supplied with an explicit provider+model target, is observation/intent metadata and must not override that target.

Products keep their own action → profile maps. Labs should prefer GenerationEngine explicit targets when the experiment fits this contract. When a lab needs provider-specific controls GenerationEngine cannot yet express, a bounded lab-only direct provider path is allowed; that path must not become a production seam.

---

## 4. Model / catalog authority

One GenerationEngine-owned catalog is the source for reusable model metadata. Product repositories must not copy pricing or capability tables.

The catalog is selection and metadata authority: models GenerationEngine may choose automatically through generic profiles, or describe authoritatively. It is **not** an execution allowlist. Explicit provider+model targets may execute through a registered provider without a catalog row. Unknown pricing and capability metadata for uncataloged targets remain unknown; they are not invented as zero.

Minimal catalog record:

```text
provider            # e.g. openai, fal
provider_model_id   # provider-native ID
capabilities        # text, structured_text, streaming_text, image, image_edit
structured_output   # yes / no
streaming           # yes / no
pricing             # dimensions + version/source identifier
availability        # available / deprecated / unknown
```

The catalog is intentionally incomplete: it contains only governed rows needed for reusable resolution/metadata. Explicit provider+model targets do not require catalog membership.

Unknown catalog fields are omitted or marked unknown. They are not invented as zero.

---

## 5. InferenceObservation

Inference-call truth is first-class. It is not a bag of JSON strings and not a product trace.

### Fields

```text
InferenceObservation
  provider              str | None
  requested_profile     str | None
  requested_model       str | None
  resolved_model        str | None
  response_model        str | None
  provider_transport    str | None     # intermediary execution route, if known
  provider_request_id   str | None
  provider_response_id  str | None
  input_tokens          int | None
  cached_input_tokens   int | None
  output_tokens         int | None
  reasoning_tokens      int | None
  cost_usd              float | None
  latency_ms            int
  retry_count           int              # transport retries; backward compatible
  transport_retry_count int | None       # same meaning as retry_count
  conformance_retry_count int            # structured repair attempts after invalid structure
  provider_attempt_count int | None      # total provider generate invocations
  state                 completed | refused | failed | incomplete
  failure_code          str | None
  pricing_source        str | None   # catalog version / identifier used for cost
```

Python names may differ; semantics must not.

### Unknown vs zero

- `None` means the provider or layer did not supply the value
- `response_model` may itself be a floating alias; `provider_transport` does not prove an upstream concrete model version
- `0` means the provider supplied zero
- missing usage must not become `0` just to satisfy a numeric field
- reasoning tokens are an additional output-token breakdown; unknown stays
  `None` and an explicit provider zero stays `0`
- `latency_ms` and `retry_count` are always known to GenerationEngine because it owns the call loop

### Latency

`latency_ms` is wall time of the GenerationEngine operation, **including retries**. It starts when the core begins the attempt loop and ends when it returns or raises its normalized result.

### Retry count

`retry_count` is the number of **additional transport** attempts after the first try of a provider call. `0` means the first attempt produced the final provider result (success or non-retryable failure). Exhausting a 3-attempt policy yields `retry_count == 2` if two retries ran, not a hard-coded `3`.

Structured generation may also issue a **conformance retry** after a successful provider call returned structurally invalid output. That is not a transport retry. `conformance_retry_count` counts those repairs. `provider_attempt_count` counts every provider generate invocation across both reasons.

OpenAI `usage.output_tokens_details.reasoning_tokens` and OpenRouter
`usage.completion_tokens_details.reasoning_tokens` populate the optional
reasoning count. Structured attempts sum it only when every provider attempt
reports it; one unknown attempt makes the aggregate `None`.

### Multiple provider calls

One GenerationEngine operation emits one observation. If a later higher-level helper issues multiple provider calls, it emits one observation per call. Products compose those into traces. GenerationEngine does not invent an Agent-turn aggregate.

### Availability on failure

Observations are produced for completed, refused, failed, and incomplete states whenever the core ran an attempt loop. Construction/configuration failures that occur before a provider call still produce an observation with `state=failed`, `provider` if known, and `failure_code` set; token/cost/request-id remain `None`.

### Retention

Observations **must not retain full prompts or full responses by default**.

Allowed diagnostic metadata:

```text
input size / message count
output size
schema name or schema hash
provider / model IDs
usage
latency
request IDs
failure codes
```

Products may retain richer traces under their own policy.

```text
GenerationEngine = inference-call truth
Product trace     = interaction / workflow truth
```

---

## 6. Failure semantics

Public failures are GenerationEngine types, not OpenAI/Fal/httpx exceptions.

| Code | Meaning | Retryable | Provider detail | Partial result |
| --- | --- | --- | --- | --- |
| `CONFIGURATION_UNAVAILABLE` | missing credentials, undeclared extra, or required config | no | no secrets; name the missing capability | no |
| `UNSUPPORTED_CAPABILITY` | profile/model/modality not in catalog or not wired | no | capability/model id | no |
| `INVALID_REQUEST` | caller-owned request failed GE validation | no | field/reason | no |
| `PROVIDER_REFUSED` | provider content-policy / refusal | no | sanitized provider message | no, unless provider also returned usable content (then `state=refused` with content + observation) |
| `PROVIDER_INCOMPLETE` | non-streaming provider response explicitly declared incomplete | no | safe provider-neutral message; IDs and usage only in observation | no partial text |
| `RATE_LIMITED` | provider 429 / quota | yes | retry-after if present | no |
| `PROVIDER_TIMEOUT` | overall inference budget exceeded | yes | timeout budget | no |
| `PROVIDER_UNAVAILABLE` | 5xx, overload, transport outage | yes | status if present | no |
| `PROVIDER_ERROR` | other provider/transport error | unknown | sanitized message | no |
| `MALFORMED_PROVIDER_RESPONSE` | unusable payload from provider | no | reason | no |
| `STRUCTURED_OUTPUT_INVALID` | GE-owned parse/schema enforcement failed | no | schema name/hash | raw text may be attached on the result, not in the observation |
| `STREAM_INCOMPLETE` | stream cancelled or ended without completion | no | reason | deltas already yielded remain yielded |
| `INTERNAL_ERROR` | unexpected core defect | no | generic message | no |

Retryable `yes` means GenerationEngine may retry according to policy. Retryable `unknown` means do not retry inside the core; surface the code and let the product decide.

`InferenceFailure.message` is a safe, non-secret public string. Provider-transport codes use stable messages and must not include SDK, HTTP, or exception text:

```text
RATE_LIMITED          "Provider rate limit exceeded."
PROVIDER_TIMEOUT      "Provider request timed out."
PROVIDER_UNAVAILABLE  "Provider is unavailable."
PROVIDER_ERROR        "Provider request failed."
PROVIDER_INCOMPLETE   "Provider returned an incomplete response."
```

Do not collapse distinct states into `INTERNAL_ERROR`.

---

## 7. Streaming

Streaming is transport-neutral.

The core must not emit HTTP/SSE framing:

```text
data: ...\n\n
[DONE]
[ERROR]
```

Public event kinds:

```text
TextDelta(text)
TextCompleted(final_text, observation)
TextFailed(failure, observation)
```

Every fully consumed stream ends with exactly one terminal event: `TextCompleted`
or `TextFailed`. Previously yielded deltas remain available after failure, but are
provisional output, not a successful or publishable artifact. Early consumer close
or task cancellation guarantees cleanup, not delivery to a retired consumer.

For schema-bearing `stream_text`, callers must set `max_conformance_retries=0`.
Invalid schemas and repair-enabled requests fail before provider construction/IO.
OpenAI forwards the native strict schema, including its name, in the streaming
request. Regardless of provider, GE parses and validates complete final text against
its captured caller schema before yielding `TextCompleted`; a mismatch yields
`STRUCTURED_OUTPUT_INVALID`, preserving reported usage/IDs without repair or a
second request. Deltas have no conformance guarantee. The existing text terminal
carries final text, not a parsed result. Products retain domain/evidence validation
and publication ownership. Caller mutation after progress begins cannot retarget
this operation's captured schema. Schema-less stream behavior is unchanged.

`stream_text()` consumes the same overall `deadline_ms` budget as non-streaming calls. It does **not** retry after output has begun. Construction/config failures, provider exceptions, overall-budget timeouts, and duplicate provider terminals are all normalized to that single public terminal:

```text
partial stream + deadline
  → exactly one TextFailed(PROVIDER_TIMEOUT)

provider/config failure before first delta
  → exactly one TextFailed

provider exception during stream
  → exactly one TextFailed

duplicate provider terminal
  → only one public terminal
```

Product backends translate those events into SSE, WebSocket, CLI, or other transports.

Legacy SSE-framing helpers are not part of the public core. Product backends translate the transport-neutral events above into their own delivery protocol.

---

## 8. Structured output

[STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md) is the adopted refinement of this section. Provider-native strict-schema features are implementation strategies, not the semantic definition of `generate_structured()`.

- Products own Pydantic/domain schemas and their domain meaning (`MapSpec`, card item schemas, and so on).
- GenerationEngine owns structural conformance of inference output to the caller-supplied schema.
- The engine accepts **JSON Schema** (current) and may later accept a Pydantic type as a convenience that is immediately reduced to JSON Schema. The public contract must not require importing product models.
- Refusal uses `PROVIDER_REFUSED`. A provider-declared incomplete non-streaming response uses `PROVIDER_INCOMPLETE` and stops before local conformance or repair. Parse/schema mismatch on completed output uses `STRUCTURED_OUTPUT_INVALID`.
- Result shape: text content, optional parsed object, observation, optional failure. Parsed data is not a product domain type inside the engine.
- Tests use a domain-neutral schema (for example a `{name: str, count: int}` fixture), never MapSpec/statblock/card models.

Current implementation:

- `generate_structured()` always performs GenerationEngine-owned local JSON Schema validation before success.
- OpenAI may submit provider-native JSON Schema as a provider-specific optimization. Adapters return raw text; GE owns parse, local validation, and bounded repair.
- OpenRouter uses ordinary chat completions plus JSON instructions. It does **not** send `response_format=json_schema`.
- One initial attempt plus at most one generic structural repair share the original `deadline_ms`.
- Labs that need provider-specific routing/reasoning knobs GE cannot express may still use a bounded direct path.

---

## 9. Image generation vs artifact persistence

GenerationEngine returns generated image content. It does not require Cloudflare, R2, or any durable store to execute image generation or editing.

`ImageRequest.options` accepts an optional strict `ImageOptions` object. Omitted
fields preserve the adapter defaults. Supported explicit fal controls are:

- GPT Image 1.5: `quality` (`low`, `medium`, `high`), `background` (`auto`,
  `transparent`, `opaque`), and `output_format` (`png`, `jpeg`, `webp`).
- FLUX 2 Pro: unsigned 32-bit `seed`, and `output_format` (`png`, `jpeg`).
- Other existing image models: no additional explicit controls yet.

Unsupported combinations fail as `INVALID_REQUEST` before a fal submission.
For GPT Image requests with options, dimensions must be one of 1024×1024,
1536×1024 or 1024×1536; they are not silently rounded. Older requests without
options retain their existing size translation. This contract does not expose
arbitrary provider parameters or promise all models support the same controls.

Returned PNG/JPEG/WebP MIME is identified from byte signatures; unrecognized
encoding is `application/octet-stream`. Actual dimensions remain unknown (`None`)
because the current provider protocol returns bytes, not verified dimensions.
Requested dimensions are not presented as measured results.

Current result shape:

```text
GeneratedImage
  content            bytes
  media_type         e.g. image/png
  width / height     when known

ImageResult
  images             list[GeneratedImage]
  observation        InferenceObservation
```

GenerationEngine returns inference content to the product; durable publication is outside the inference core.

Topology:

```text
GenerationEngine
   ↓ generated image result
Product backend
   ↓ product-owned artifact policy
Cloudflare / R2 / etc.
```

`GenerationClient.generate_image()` / `edit_image()` return `GeneratedImage` bytes. DungeonMindServer owns Cloudflare persistence.

Image persistence helpers are not part of the inference core. `UploadService` is deleted.

---

## 10. Credentials and optional capabilities

A text-only consumer must be able to construct and call text generation without Fal or Cloudflare credentials or packages.

A Fal image consumer must fail with `CONFIGURATION_UNAVAILABLE` / `UNSUPPORTED_CAPABILITY` when Fal is requested without the extra or credentials.

Recommended packaging:

```text
core:             pydantic, httpx, tenacity, jsonschema
openai extra:     openai
openrouter extra: openai   # OpenAI-compatible SDK; provider identity remains openrouter
fal extra:        fal-client
dev group:        pytest, pytest-asyncio, ruff
```

`GenerationClient.from_env()` lazy-loads OpenAI, OpenRouter, and Fal adapters on first use so a core-only wheel import does not require provider extras. CI proves that boundary with an isolated built-wheel import step.

Text through OpenAI requires `OPENAI_API_KEY`. Text through OpenRouter requires `OPENROUTER_API_KEY`. Images require `FAL_KEY`.

Cloudflare is not a GenerationEngine inference dependency.

Do not create separate provider packages in E2 unless the extras model proves insufficient.

---

## 11. Cutover policy

E2B intentionally did not replace DungeonMindServer imports because that
product was outside the foundation slice.

The coordinated E2 cutover uses this contract for GenerationEngine and its
DungeonMindServer consumers. Obsolete GenerationEngine surfaces are deleted,
not retained as compatibility paths. See [COMPATIBILITY.md](COMPATIBILITY.md).

Do not add deprecation frameworks, dual APIs, or SSE/URL adapters to stretch old surfaces past that cutover.

---

## Out of scope for the core contract

```text
product action mappings
DungeonMind knowledge
Buddy Agent turns / traces as a GE type
durable artifact ownership
HTTP API for GenerationEngine
moderation as an undeclared extra capability
ecosystem-wide model policy copied from products
```


## Opt-in decision dispatch authorization

`generationengine.decision_dispatch` supplies `DecisionDispatchGate`,
`PreparedDecisionDispatch` and `DecisionDispatchEvent`. A provider that supports
this optional gate exposes its actual SDK-serialized, uncompressed request body
as immutable bytes, SHA-256 and size, together with attempt ID, provider,
transport, model, HTTP method and exact destination. Secret headers are excluded.
Body bytes may contain private caller data; storage and authorization are owned
by the calling product. This is a transport entity, not the entire HTTP wire or
an estimate of provider token usage.

The TypeSafe adapter accepts `dispatch_gate=...`. The official SDK serializes the
request before the product's async `authorize(request)` callback runs. Only an
explicit `True` permits progress. The product must also acknowledge
`record(event)` with explicit `True`. The lifecycle is:

- `sending`: acknowledged before entering the underlying HTTP transport;
- `known_not_sent`: authorization, pre-send recording or request-integrity failure;
- `response_received`: HTTP response headers/status returned, before SDK parsing;
- `outcome_unknown`: transport entry occurred but no response was acknowledged.

`response_received` does not mean a valid decision or complete response body.
A timeout/cancellation after transport entry is conservatively unknown even if
no bytes ultimately left the machine. Callback failures propagate as failed
execution; a failed terminal record does not establish a durable terminal event.
The product must reconcile its pending records and must not replay an uncertain
request. GE does not persist events or grant authorization itself.

Guarded TypeSafe execution disables SDK compression/retries, HTTP redirects and
transport retries, and never automatically retries through GE, even when the
request otherwise allows transport retries. The product selects and validates
its permitted model and scope from the actual prepared request. Concurrent
requests have distinct attempt IDs and independent lifecycle records. Closing
the owning provider closes the injected transport. An injected SDK client cannot
be combined with a dispatch gate. Unguarded provider behavior is unchanged.


## Opt-in text dispatch authorization

`generationengine.text_dispatch` supplies `TextDispatchGate`,
`PreparedTextDispatch` and `TextDispatchEvent`. The OpenAI text adapter accepts
`dispatch_gate=...` for ordinary nonstreaming and streaming Responses calls. The actual SDK
serializes the body before consumer authorization. Immutable bytes, SHA-256,
size, UUID attempt ID, provider, API mode, model, method and exact destination
are exposed; authentication headers are not. Body bytes may contain caller data.

This bounded mode sends POST `https://api.openai.com/v1/responses` with explicit
`tools=[]` and an explicit boolean `stream` (`false` for ordinary calls, `true`
for streaming). It pins that destination instead of inheriting an
environment base URL. SDK and GE transport retries, redirects and ambient proxy
routing are disabled. The consumer can inject an HTTP transport for testing but
cannot combine an injected SDK client with a gate. Default execution is unchanged.
Repair-enabled JSON-schema calls and JSON-object streaming are rejected before
provider I/O.
Strict schema calls are supported with explicit `max_conformance_retries=0`;
`generate_structured` still validates the returned result locally and fails without
a second request on nonconformance. The actual strict schema is part of the
authorized/captured SDK envelope. Ordinary JSON-object mode remains supported.

Authorization and lifecycle callbacks must return explicit `True`. Lifecycle
uses the same meanings as the decision gate: `sending` before underlying
transport entry; `known_not_sent` for pre-entry failure; `outcome_unknown` after
entry failure/cancellation; `response_received` when HTTP headers/status return,
before body completion or SDK parsing. A malformed/incomplete body does not
undo a response-header acknowledgment. Failed terminal acknowledgment is not a
durable terminal record. Products reconcile pending/unknown attempts and own
subsequent authorization; this gate supplies no permissions or persistence.
Guarded text success additionally requires SDK response status `completed`, no
response error, and nonblank output text. An HTTP200 response with failed,
nonterminal, missing or unknown status cannot become a successful text result,
even when it contains residual valid text. Failure observations preserve safe
provider IDs/model and reported usage, without residual text or raw error detail.
Guarded streaming uses the same exact-byte authorization and header lifecycle.
Only a genuine `response.completed` event with completed status, no error/refusal
and nonblank final text can succeed. Failed/incomplete/cancelled events, unknown
status, missing terminal, malformed SSE/body and body transport errors cannot
promote earlier deltas to success. Missing terminal uses `STREAM_INCOMPLETE`;
provider-declared incomplete uses `PROVIDER_INCOMPLETE`. Safe IDs/model and reported
usage remain available when known, including the HTTP request ID after headers.
An SDK-wrapped typed gate denial remains an evidence-persistence failure. Header
acknowledgment is not undone by body failure. Response streams close on terminal,
early close, timeout, cancellation and parsing/callback failure. No stream retry.
Closing the provider closes its owned SDK client and injected HTTP transport.

The supported OpenAI SDK `http_client` injection and `max_retries=0` are documented
in the [official Python reference](https://developers.openai.com/api/reference/python).
Tests exercise the installed locked SDK 2.14.0 with HTTPX; this slice does not
upgrade the SDK to the newer HTTPX2 documentation implementation.
