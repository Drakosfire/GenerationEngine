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
```

Do not add methods named `generate_statblock`, `generate_card`, `generate_map`, or `generate_character`.

Embeddings, transcription, speech, and moderation are out of this contract until an explicit ownership decision adds them.

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
  provider_request_id   str | None
  provider_response_id  str | None
  input_tokens          int | None
  cached_input_tokens   int | None
  output_tokens         int | None
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
- `0` means the provider supplied zero
- missing usage must not become `0` just to satisfy a numeric field
- `latency_ms` and `retry_count` are always known to GenerationEngine because it owns the call loop

### Latency

`latency_ms` is wall time of the GenerationEngine operation, **including retries**. It starts when the core begins the attempt loop and ends when it returns or raises its normalized result.

### Retry count

`retry_count` is the number of **additional transport** attempts after the first try of a provider call. `0` means the first attempt produced the final provider result (success or non-retryable failure). Exhausting a 3-attempt policy yields `retry_count == 2` if two retries ran, not a hard-coded `3`.

Structured generation may also issue a **conformance retry** after a successful provider call returned structurally invalid output. That is not a transport retry. `conformance_retry_count` counts those repairs. `provider_attempt_count` counts every provider generate invocation across both reasons.

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

Target event kinds:

```text
TextDelta(text)
TextCompleted(final_text, observation)
TextFailed(failure, observation)
```

Every stream must end with exactly one terminal event: `TextCompleted` or `TextFailed`. Partial deltas remain valid when a stream ends in `TextFailed`.

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

E2A found no DungeonMindServer caller of legacy `generate_stream`. The coordinated cutover replaces SSE framing with these transport-neutral events and deletes the old streaming surface.

---

## 8. Structured output

[STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md) is the adopted refinement of this section. Provider-native strict-schema features are implementation strategies, not the semantic definition of `generate_structured()`.

- Products own Pydantic/domain schemas and their domain meaning (`MapSpec`, card item schemas, and so on).
- GenerationEngine owns structural conformance of inference output to the caller-supplied schema.
- The engine accepts **JSON Schema** (current) and may later accept a Pydantic type as a convenience that is immediately reduced to JSON Schema. The public contract must not require importing product models.
- Refusal uses `PROVIDER_REFUSED`. Parse/schema mismatch uses `STRUCTURED_OUTPUT_INVALID`.
- Result shape: text content, optional parsed object, observation, optional failure. Parsed data is not a product domain type inside the engine.
- Tests use a domain-neutral schema (for example a `{name: str, count: int}` fixture), never MapSpec/statblock/card models.

Current E5B.1 implementation:

- `generate_structured()` always performs GenerationEngine-owned local JSON Schema validation before success.
- OpenAI may submit provider-native JSON Schema as a provider-specific optimization. Adapters return raw text; GE owns parse, local validation, and bounded repair.
- OpenRouter uses ordinary chat completions plus JSON instructions. It does **not** send `response_format=json_schema`.
- One initial attempt plus at most one generic structural repair share the original `deadline_ms`.
- Labs that need provider-specific routing/reasoning knobs GE cannot express may still use a bounded direct path.

---

## 9. Image generation vs artifact persistence

GenerationEngine returns generated image content. It does not require Cloudflare, R2, or any durable store to execute image generation or editing.

Target result:

```text
GeneratedImage
  content            bytes | controlled temporary / provider reference
  media_type         e.g. image/png
  width / height     when known
  observation        InferenceObservation
```

Exact bytes-vs-reference representation may account for memory cost, but durable publication is outside the inference core.

Target topology:

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
