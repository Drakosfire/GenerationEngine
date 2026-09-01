# Compatibility surfaces

Transition-before-demolition applies to **demonstrated consumer seams**, not every historical package export.

```text
CONSUMER COMPATIBILITY
  An actual DungeonMindServer import or behavioral shape.
  Must survive until that consumer migrates (E3).

BASELINE PUBLIC INVENTORY
  Historical generationengine.__all__ / unused APIs.
  Characterized here. May be retired or replaced during E2
  when no real consumer depends on it.
```

Adding target types (`InferenceObservation`, `TextProvider`, catalog types) is allowed in E2.

---

## Consumer compatibility

These are the DungeonMindServer seams from [CURRENT-STATE.md](CURRENT-STATE.md). E2 must not break them. E3 owns migration.

### Package-root imports

| Symbol | Why it is a seam |
| --- | --- |
| `ImageService` | Map, inpaint, image-management, card image paths |
| `MetricsService` | Card constructs it and passes it into `ImageService` |
| `ImageGenerationRequest` | Image callers |
| `ImageModel` | Image callers and `shared/image_models.MODEL_MAP` |
| `ImageSize` | Image callers |
| `TextGenerationService` | MapSpec, SVG mask (package root) |
| `TextGenerationRequest` | MapSpec, SVG mask |
| `TextModel` | MapSpec, SVG mask, PCG |

### Internal import paths

| Import path | Consumer |
| --- | --- |
| `generationengine.services.text_service.TextGenerationService` | PCG; dormant RulesLawyer import |
| `generationengine.models.requests.TextGenerationRequest` | PCG; dormant RulesLawyer import |
| `generationengine.models.requests.TextModel` | PCG; dormant RulesLawyer import |
| `generationengine.models.image_responses.ImageGenerationResponse` | Server tests |
| `generationengine.models.image_responses.ImageResult` | Server tests; `.url` is the Cloudflare URL assumption |

Dormant RulesLawyer imports are still import-path compatibility until that file drops them. They are not a reason to preserve unused **other** exports.

### Return / behavioral shapes

Callers use these attributes. Facades may wrap new internals as long as these keep working:

```text
TextGenerationService.generate(request, service_name=...) ->
  success
  content
  parsed_content
  error.message
  metrics.tokens_used

ImageService.generate(request) ->
  success
  images[].url
  images[].width / height
  error.message
  metrics.duration_ms
  metrics.model_used
  metrics.retry_count

TextModel.GPT_5_1
ImageModel.{FLUX_2_PRO, NANO_BANANA_PRO, GPT_IMAGE_15, FLUX_PRO, FLUX_LORA_I2I}
ImageSize.{SQUARE, PORTRAIT, LANDSCAPE}
ImageService() constructor currently requires Cloudflare (until E2D facade)
TextGenerationService() constructor currently requires OPENAI_API_KEY
```

`TextGenerationResponse`, `ImageGenerationResponse`, `ImageResult`, `GenerationError`, and `GenerationMetrics` are consumer return types even when not imported at package root.

### Behavioral facades that have consumers

| Current behavior | Until | Successor |
| --- | --- | --- |
| `TextGenerationService` constructs `AsyncOpenAI` | E2C (class/signature stay) | `TextProvider` behind the same facade |
| `ImageService.generate` uploads to Cloudflare and returns URLs | E3 | E2D generation result + URL-returning adapter |
| `UploadService` required to construct `ImageService` | E2D | optional persistence; keep a URL facade for Server |
| `TextModel.GPT_5_1` as the live text selector | E3 unless a facade test proves otherwise | catalog + profiles |
| `MODEL_PRICING` in `text_service.py` | E2C | GE catalog; effective cost for `gpt-5.1` callers may change once catalog is truthful |

---

## Baseline public inventory

Snapshot of `generationengine.__all__` at E2A baseline. This is characterization, **not** an E2 freeze.

| Symbol | Consumer? | E2 note |
| --- | --- | --- |
| `IGenerator` | no | Product-tainted (`statblock`/`card`/`character`/`store`). E2B may retire or quarantine it. |
| `GenerationResponse` | no | Unused generic wrapper. E2 may retire. |
| `GenerationError` | return shape | Keep while consumer responses expose `.error.message`. |
| `GenerationMetrics` | return shape | Keep while callers read duration/model/retries/tokens. Stop putting prompts in `input` during E2. |
| `ErrorCode` | no direct import | May move with failure taxonomy; do not require E3 to change unused enum members. |
| `is_retryable` | no | Historical helper. E2 may retire from public exports. |
| `ImageGenerationRequest` | yes | Consumer compatibility. |
| `ImageModel` | yes | Consumer compatibility. |
| `ImageSize` | yes | Consumer compatibility. |
| `TextGenerationRequest` | yes | Consumer compatibility. |
| `TextGenerationResponse` | return shape | Consumer compatibility. |
| `TextModel` | yes | Consumer compatibility. |
| `ImageProvider` | no Server import | Target protocol. Public export may move; keep the protocol in core. |
| `ImageService` | yes | Consumer compatibility. |
| `MetricsService` | yes | Consumer compatibility. |
| `TextGenerationService` | yes | Consumer compatibility. |
| `make_schema_strict` | no Server import | Target mechanics; may drop from `__all__` during E2 if unused. |
| `RetryableError` | no | Leaked retry exception. E2 may stop exporting it. |

### Baseline behaviors with no consumer

| Current behavior | E2 note |
| --- | --- |
| `IGenerator.generator_type` product IDs | E2B may remove the protocol. No E3 gate. |
| `TextGenerationService.generate_stream` SSE / `[DONE]` / `[ERROR]` | No DungeonMindServer caller. E2C may replace with transport-neutral events. An SSE adapter is optional, not a compatibility obligation. |
| `OpenAIImageProvider` unwired | E2D owns truthful wiring. Not a consumer freeze. |

---

## What E2 must not do

- migrate DungeonMindServer or Buddy callers
- break the consumer import paths or return shapes listed above
- change live Fal model IDs used by current image callers merely to “fix” the catalog
- wire `OpenAIImageProvider` into production `ImageService` in E2A (E2D owns truthful wiring)
- remove Cloudflare upload from the **consumer** `ImageService.generate` path before a URL-returning facade exists
- add product profile names to the core
- publish a network API
- treat unused `__all__` entries as E3-protected compatibility

---

## E3 migration seams (recorded, not executed)

| Consumer | Seam |
| --- | --- |
| MapSpec structured text | Keep product schema/prompts; switch to profile `structured_high_reliability` + observation |
| SVG mask text | Unstructured text / `text_fast` or structured SVG schema owned by the product |
| PCG text | Same; stop importing internal `services.text_service` modules |
| Map / inpaint / image management / card images | Product persists URLs; GE returns image bytes or a documented compatibility URL adapter |
| Card item description | Today direct OpenAI `gpt-4o`. First simple E3 text-migration candidate |
| RulesLawyer | Dormant GE imports; live path is direct `AsyncOpenAI`. Not an E2 consumer |
