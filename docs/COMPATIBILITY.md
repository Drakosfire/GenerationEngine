# Cutover inventory

GenerationEngine is a single-owner system. There is no long-lived compatibility architecture.

```text
E2B
  make GE trustworthy
  land core primitives
  remove obviously dead API
  do not intentionally break active DMS imports
      only because DMS is not changing in this PR

NEXT CUTOVER
  change GenerationEngine + DungeonMindServer together
  move every known consumer
  prove the replacement
  delete the old GE architecture immediately
```

This file is **consumer inventory / cutover evidence**, not an API-support promise.

---

## Active DungeonMindServer seams (sequencing constraint during E2B)

These paths remain importable in E2B because DungeonMindServer is not in this PR.

### Package-root imports

`ImageService`, `MetricsService`, `ImageGenerationRequest`, `ImageModel`, `ImageSize`, `TextGenerationService`, `TextGenerationRequest`, `TextModel`

### Internal import paths

```text
generationengine.services.text_service.TextGenerationService
generationengine.models.requests.TextGenerationRequest
generationengine.models.requests.TextModel
generationengine.models.image_responses.ImageGenerationResponse
generationengine.models.image_responses.ImageResult
```

### Return shapes callers read

```text
success, content, parsed_content, error.message
images[].url / width / height
metrics.tokens_used / duration_ms / model_used / retry_count
TextModel.GPT_5_1
ImageModel.{FLUX_2_PRO, NANO_BANANA_PRO, GPT_IMAGE_15, FLUX_PRO, FLUX_LORA_I2I}
```

Do not build adapters, deprecation layers, dual APIs, SSE adapters, or URL-returning shims to keep these alive after the cutover.

---

## Removed in E2B (no current consumer)

Search of DungeonMindServer `main` found no imports of these symbols. They are gone from the package root:

| Symbol | Evidence |
| --- | --- |
| `IGenerator` | only defined in GenerationEngine; product-tainted `generator_type` |
| `GenerationResponse` | unused generic wrapper; Server uses text/image response types |
| `RetryableError` public export | internal retry exception; Server does not import it |
| `is_retryable` public export | internal helper; Server does not import it |
| `make_schema_strict` public export | still used internally by `TextGenerationService` |

`ErrorCode` / `GenerationError` remain because current response objects still expose them.

---

## Next cutover deletes (not E2B)

```text
legacy TextGenerationService / ImageService facades after consumers move
GenerationMetrics prompt-era fields once observations are populated
ErrorCode once FailureCode is on the live path
SSE generate_stream
MODEL_PRICING / TextModel as authority
UploadService in the inference core
```
