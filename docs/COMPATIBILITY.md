# Compatibility surfaces until E3

**Rule:** DungeonMindServer imports remain until E3 migrates them. E2A does not delete or rename these symbols to match the target contract.

Labels:

```text
KEEP AS TARGET                 remains with the same role in the new contract
KEEP TEMPORARILY / COMPATIBILITY
                               keep working; wrap or adapt during E2
REPLACE IN E2                  new type/path supersedes internals; facade stays
REMOVE ONLY AFTER E3           delete only after real consumers have moved
```

---

## Package-root exports (`generationengine.__all__`)

Current inventory from baseline `__init__.py`:

| Symbol | Label | Notes |
| --- | --- | --- |
| `IGenerator` | REPLACE IN E2 | Protocol is unused by Server. `generator_type` documents product IDs. Replacement must not carry product vocabulary. Keep export until unused. |
| `GenerationResponse` | KEEP TEMPORARILY | Generic wrapper; Server does not import it today. |
| `GenerationError` | KEEP TEMPORARILY | Shape is useful; codes expand in E2B. |
| `GenerationMetrics` | REPLACE IN E2 | Superseded by `InferenceObservation`. Keep as compatibility field on current responses. Stop putting prompts in `input`. |
| `ErrorCode` | KEEP TEMPORARILY | Expand; do not remove existing members until E3. |
| `is_retryable` | KEEP TEMPORARILY | Keep beside expanded failure taxonomy. |
| `ImageGenerationRequest` | KEEP TEMPORARILY | Active Server import. New image request types may be added alongside. |
| `ImageModel` | KEEP TEMPORARILY | Active Server import / `MODEL_MAP`. Catalog supersedes as authority; enum remains a compatibility selector. |
| `ImageSize` | KEEP TEMPORARILY | Active Server import. |
| `TextGenerationRequest` | KEEP TEMPORARILY | Active Server import. Profile field may be added later without removing `model`. |
| `TextGenerationResponse` | KEEP TEMPORARILY | Active via generate() return even when imported from internal modules. |
| `TextModel` | KEEP TEMPORARILY | Active Server import. Stale single-member enum; do not silently expand product policy into it. Catalog is the authority. |
| `ImageProvider` | KEEP AS TARGET | Protocol stays; implementation wiring is E2D. |
| `ImageService` | KEEP TEMPORARILY | Active Server singleton. E2D may add a generation-only API and keep this as URL-returning facade. |
| `MetricsService` | KEEP TEMPORARILY | Card constructs it. In-memory stub. Not observation authority. |
| `TextGenerationService` | KEEP TEMPORARILY | Active Server import (root and internal path). E2C facade over `TextProvider`. |
| `make_schema_strict` | KEEP AS TARGET | Structured-output mechanics belong in GE. |
| `RetryableError` | KEEP TEMPORARILY | Internal retry exception leaked publicly. Normalize in E2B/E2C. |

Not in `__all__` but imported by Server:

| Import path | Label | Consumer |
| --- | --- | --- |
| `generationengine.services.text_service.TextGenerationService` | KEEP TEMPORARILY | PCG, dormant RulesLawyer import |
| `generationengine.models.requests.TextGenerationRequest` | KEEP TEMPORARILY | PCG, dormant RulesLawyer import |
| `generationengine.models.requests.TextModel` | KEEP TEMPORARILY | PCG, dormant RulesLawyer import |
| `generationengine.models.image_responses.ImageGenerationResponse` | KEEP TEMPORARILY | Server tests |
| `generationengine.models.image_responses.ImageResult` | KEEP TEMPORARILY | Server tests; `.url` is the Cloudflare assumption |

Adding target types (`InferenceObservation`, `TextProvider`, catalog types) is allowed. Removing the rows above is not allowed in E2.

---

## Behavioral facades

| Current behavior | Label | Successor |
| --- | --- | --- |
| `TextGenerationService` constructs `AsyncOpenAI` | REPLACE IN E2 | E2C `TextProvider` |
| `generate_stream` yields SSE / `[DONE]` / `[ERROR]` | REMOVE ONLY AFTER E3 | E2C transport-neutral events + optional SSE adapter |
| `ImageService` registers only Fal | REPLACE IN E2 | E2D truthful wiring |
| `ImageService.generate` uploads to Cloudflare and returns URLs | REMOVE ONLY AFTER E3 | E2D generation result + compatibility URL adapter |
| `UploadService` required to construct `ImageService` | REPLACE IN E2 | E2D optional persistence |
| `TextModel.GPT_5_1` only | KEEP TEMPORARILY | Catalog + profiles; do not turn the enum into product policy |
| `MODEL_PRICING` in `text_service.py` | REPLACE IN E2 | GE catalog/pricing authority |
| `IGenerator.generator_type` product IDs | REPLACE IN E2 | E2B drop product vocabulary from new core modules; keep old protocol until unused |
| Full prompt in `GenerationMetrics.input` | REPLACE IN E2 | Observation retention rule |

---

## What E2 must not do

- migrate DungeonMindServer or Buddy callers
- change live provider/model IDs used by current facades merely to “fix” the catalog
- wire `OpenAIImageProvider` into production `ImageService` in E2A (E2D owns truthful wiring)
- remove Cloudflare upload from the current `ImageService.generate` path before a replacement facade exists
- add product profile names to the core
- publish a network API

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
