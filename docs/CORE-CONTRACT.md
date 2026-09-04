# GenerationEngine core contract (target)

**Status:** Coordinated cutover implemented. Live execution is `GenerationClient` over OpenAI and Fal adapters.  
**Current behavior:** [CURRENT-STATE.md](CURRENT-STATE.md)  
**Cutover inventory:** [COMPATIBILITY.md](COMPATIBILITY.md)

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

Hosting remains in-process. A network GenerationEngine service is not part of E2.

---

## 2. Provider boundary

Capability-focused protocols, not one giant provider type:

```text
TextProvider
ImageProvider
```

After the provider reset:

- the core/service layer must not instantiate OpenAI or Fal SDK clients
- provider SDK exception types must not be the public contract
- advertised providers must match registered wiring and declared extras

`ImageProvider` already exists as a protocol. E2B adds `TextProvider`. The coordinated cutover moves live OpenAI/Fal execution behind these seams.

---

## 3. Model selection boundary

```text
product action                 # owned by the consumer
        ↓
generic inference profile      # GenerationEngine vocabulary
        ↓
provider + model resolution    # GenerationEngine-owned
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

Explicit provider/model selection remains allowed for tests and for compatibility facades. Precedence:

1. explicit provider + model (test/compatibility override)
2. generic profile resolution through the catalog
3. no implicit product-policy fallback inside GenerationEngine

Products keep their own action → profile maps.

---

## 4. Model / catalog authority

One GenerationEngine-owned catalog is the source for reusable model metadata. Product repositories must not copy pricing or capability tables.

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

E2A does not populate a complete production catalog. E2B lands the authority and shape. Later slices fill rows required by wired providers.

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
  retry_count           int
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

`retry_count` is the number of **additional** attempts after the first try. `0` means the first attempt produced the final result (success or non-retryable failure). Exhausting a 3-attempt policy yields `retry_count == 2` if two retries ran, not a hard-coded `3`.

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

Every stream must end with exactly one terminal event: `TextCompleted` or `TextFailed`. Partial deltas remain valid when a stream ends in `TextFailed` with `STREAM_INCOMPLETE`.

Product backends translate those events into SSE, WebSocket, CLI, or other transports.

E2A found no DungeonMindServer caller of legacy `generate_stream`. The coordinated cutover replaces SSE framing with these transport-neutral events and deletes the old streaming surface.

---

## 8. Structured output

- Products own Pydantic/domain schemas (`MapSpec`, card item schemas, and so on).
- GenerationEngine owns provider mechanics: JSON Schema submission, strict-mode adaptation, reporting parse/refusal outcomes.
- The engine accepts **JSON Schema** (current) and may later accept a Pydantic type as a convenience that is immediately reduced to JSON Schema. The public contract must not require importing product models.
- Schema normalization (`additionalProperties`, required fields, `$ref` cleaning) lives in GenerationEngine.
- Refusal uses `PROVIDER_REFUSED`. Parse/schema mismatch uses `STRUCTURED_OUTPUT_INVALID`.
- Result shape: text content, optional parsed object, observation, optional failure. Parsed data is not a product domain type inside the engine.
- Tests use a domain-neutral schema (for example a `{name: str, count: int}` fixture), never MapSpec/statblock/card models.

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
core:         pydantic, httpx, tenacity
openai extra: openai
fal extra:    fal-client
dev group:    pytest, pytest-asyncio, ruff
```

`GenerationClient.from_env()` lazy-loads OpenAI and Fal adapters on first use so a core-only wheel import does not require provider extras. CI proves that boundary with an isolated built-wheel import step.

Cloudflare is not a GenerationEngine inference dependency.

Do not create separate provider packages in E2 unless the extras model proves insufficient.

---

## 11. Cutover policy

E2B does not replace current DungeonMindServer imports because that product is not in this PR.

The next slice moves GenerationEngine and DungeonMindServer together, then deletes obsolete GE surfaces immediately. See [COMPATIBILITY.md](COMPATIBILITY.md).

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
