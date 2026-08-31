# GenerationEngine current state (E2A characterization)

**Baseline:** `f6f0fa95c745e4201785f3cf1d29d853e603a592` (`feat(GenerationEngine): add inpainting and mask support`, 2026-01-09)  
**Characterization date:** 2026-08-31  
**This document describes implemented behavior.** The target contract is [CORE-CONTRACT.md](CORE-CONTRACT.md). Do not treat this file as desired architecture.

---

## Baseline environment

Recorded from a clean worktree of that commit:

```text
python (system): 3.12.3
uv: 0.5.20
uv selected interpreter: CPython 3.13.1
uv lock --check: pass
uv sync --all-groups: pass (24 packages)
uv build: pass (sdist + wheel)
GitHub Actions on main: absent (no .github/workflows)
tracked __pycache__ / .pyc: present (34 bytecode files)
.gitignore: absent on baseline
```

Clean-install package identity:

```text
distribution: generationengine==0.1.0
openai: installed (2.14.0, required dependency)
fal-client: NOT INSTALLED (undeclared)
ruff: declared as extra == 'dev', not installed by `uv sync --all-groups`
pytest: installed because it is a runtime dependency (packaging smell)
```

`uv sync --all-groups` installs `[dependency-groups] dev` (`pytest-asyncio` only). It does not install `[project.optional-dependencies] dev` (`ruff`, older pytest pins). `uv run ruff check .` therefore fails with `Failed to spawn: ruff` after a default sync.

---

## Claimed vs implemented

| README / type claim | Implemented today |
| --- | --- |
| Image generation via Fal.ai **and** OpenAI DALL-E | `OpenAIImageProvider` exists. `ImageService` registers only `FalProvider` against Fal model IDs. OpenAI image provider is unwired. |
| Text generation via OpenAI Responses API | True. `TextGenerationService` constructs `AsyncOpenAI` directly. No `TextProvider` protocol. |
| README example uses `TextModel.GPT_4O` | `TextModel` has only `GPT_5_1 = "gpt-5.1"`. The documented example is stale. |
| Cloudflare credentials optional | `ImageService()` always constructs `UploadService()` unless injected. `UploadService` requires `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_IMAGES_API_TOKEN` at init. Image construction therefore requires Cloudflare even when no image call is made. |
| Fal.ai optional | `FalProvider` imports `fal_client` and raises `ImportError` if missing. `fal-client` is not a package dependency. ImageService swallows Fal init failure and continues with an empty provider map, then still requires Cloudflare. |
| Metrics tracking | `GenerationMetrics` plus in-memory `MetricsService` stub. Full system prompt (text) / full image prompt stored in `input`. No provider, request ID, cached tokens, response model, or normalized failure state. |
| Robust retry with exponential backoff | `retry_with_backoff` exists (3 attempts, 1s/2s/4s). Text success path never records actual retry count (`retry_count` stays `0`, or is hard-coded to `3` after exhaustion). |
| `IGenerator.generator_type` documents `statblock`, `card`, `character`, `store` | Product vocabulary in the shared protocol. No production `IGenerator` implementer in this package. |

---

## Capability truth

### Text

```text
path: TextGenerationService → AsyncOpenAI.responses.create
provider protocol: none
models: TextModel.GPT_5_1 only
pricing: MODEL_PRICING["gpt-5.1"] approximate 2024-era $0.005/$0.015 per 1K tokens
max_tokens: accepted on the request model, explicitly ignored (Responses API)
schema: JSON Schema dict; make_schema_strict() adapts for OpenAI strict mode
parse failure: JSONDecodeError is logged; response remains success=True with parsed_content=None
refusal: mapped to ErrorCode.PROVIDER_REJECTED, success=False
constructor: ValueError if OPENAI_API_KEY missing
```

### Streaming text

```text
path: TextGenerationService.generate_stream → responses.stream
output: SSE strings (`data: {delta}\n\n`, `data: [DONE]\n\n`, `data: [ERROR]...\n\n`)
structured schema: ignored with a warning
retries/metrics/observation: none
DungeonMindServer consumers of generate_stream: none found
```

### Image

```text
wired providers: FalProvider only, keyed as
  flux-2-pro, nano-banana-pro, gpt-image-1.5, flux-pro, flux-lora-i2i
unwired: OpenAIImageProvider
result: ImageResult.url is documented as a Cloudflare URL; bytes never leave the service
persistence: generate() always uploads via UploadService before returning
constructor: ImageService() requires Cloudflare credentials
missing Fal: providers dict stays empty; generate() then returns INVALID_INPUT for any model
```

`gpt-image-1.5` in this package is a **Fal** model ID (`fal-ai/gpt-image-1.5`), not the unwired OpenAI Images provider.

### Observation / metrics

`GenerationMetrics` fields today:

```text
duration_ms, tokens_used, estimated_cost_usd, model_used,
retry_count, timestamp, input, output
```

Missing relative to the target `InferenceObservation`:

```text
provider
requested_profile
requested_model vs resolved_model vs response_model
provider_request_id
input_tokens / cached_input_tokens / output_tokens as first-class fields
unknown vs zero
normalized completion/failure state
failure_code
```

Retention:

- text `input` JSON includes the **full system prompt** and user-prompt **length**
- image `input` JSON includes the **full prompt**
- text `output` JSON includes content length and token counts (when serializable)

MagicMock usage objects make `json.dumps` of token fields raise, and the service collapses that into `INTERNAL_ERROR`. That is why three text-service tests fail on a clean baseline (see Tests below).

---

## Failure behavior today

Current `ErrorCode` values:

```text
retryable:    PROVIDER_TIMEOUT, PROVIDER_OVERLOADED, RATE_LIMITED
not retryable: INVALID_INPUT, AUTHENTICATION_REQUIRED, NOT_FOUND,
               PROVIDER_REJECTED, INTERNAL_ERROR
```

Gaps:

- missing credentials raise `ValueError` at construction, not a generation error
- unknown image model → `INVALID_INPUT`
- OpenAI `APIError` with status 429/5xx → `PROVIDER_OVERLOADED`; other `APIError` → `PROVIDER_REJECTED` wrapped as `RetryableError` (so it is retried even when marked non-retryable conceptually)
- unexpected exceptions, including metrics JSON failures, → `INTERNAL_ERROR`
- structured parse failure is not a failure
- stream errors are sentinel strings, not `GenerationError`
- provider SDK types are used internally; public methods usually wrap them, but construction still raises raw `ValueError` / `ImportError`

---

## Packaging and repository hygiene

```text
[project].name = generationengine
requires-python = >=3.11
runtime deps: pydantic, httpx, tenacity, openai, pytest
optional extra `dev`: pytest, pytest-asyncio, ruff
dependency-group `dev`: pytest-asyncio only
fal-client: undeclared
.gitignore: none on baseline
tracked bytecode: src/**/__pycache__ and tests/__pycache__
test_metrics_service.py: missing source; bytecode still tracked
wheel: hatchling omits bytecode (24 .py files + dist-info); install artifact is clean
CI: none
```

`pytest` as a runtime dependency and missing `fal-client` mean a consumer environment can look healthier or worse than a clean GenerationEngine install, depending on which extra packages the consumer already has. DungeonMindServer currently declares `fal-client` and `openai` itself.

---

## Tests on baseline

```text
uv run pytest -q
4 failed, 45 passed, 2 warnings
```

Failures (pre-existing; not fixed in E2A):

1. `test_openai_provider_generate_rate_limit` — `openai.RateLimitError` now requires `response` and `body`.
2. `test_text_generation_success`
3. `test_text_generation_with_system_prompt`
4. `test_text_generation_with_parameters`

(2)–(4) fail because the mock `usage` object is a `MagicMock`, `getattr(usage, "input_tokens", 0)` returns another `MagicMock`, and `json.dumps` of metrics `output` raises `TypeError`, mapped to `INTERNAL_ERROR`.

Pydantic deprecation warnings: class-based `Config` / `json_encoders` on `GenerationMetrics`.

These are **audit facts**, not target invariants. E2B owns making the baseline test/CI story trustworthy.

---

## Consumer inventory

Inventory is grounded in DungeonMindServer `main` at `69f57c72a55ac2679780aa7692f3abd035634a63` (E1A). Buddy has **no** `generationengine` imports.

Classification key:

```text
ACTIVE_RUNTIME         called on product request paths
ACTIVE_PRODUCT_TOOLING product scripts/tools
TEST_ONLY              tests only
DORMANT                imported, not called
HISTORICAL_DOC         docs/comments only
```

### DungeonMindServer consumers

| Consumer | Class | Capability | Symbols | Provider / model assumption | Persistence / failure / telemetry |
| --- | --- | --- | --- | --- | --- |
| `mapgenerator/prompt_compiler.py` `generate_mapspec` | ACTIVE_RUNTIME | structured text | `TextGenerationService`, `TextGenerationRequest`, `TextModel` (package root) | OpenAI via GE; `TextModel.GPT_5_1`; product-owned `MapSpec` JSON schema; `max_tokens=800` (ignored) | Uses `success`, `error.message`, `parsed_content` then `content`. Product validates `MapSpec`. No metrics consumed. |
| `mapgenerator/svg_mask.py` | ACTIVE_RUNTIME | unstructured text | same package-root text symbols | `GPT_5_1`; `max_tokens=2000` (ignored) | `success` / `content`; product extracts SVG. No metrics. |
| `routers/map_router.py` generate path | ACTIVE_RUNTIME | image generation | `ImageService`, `ImageGenerationRequest`, `ImageModel`, `ImageSize` | `ImageModel.GPT_IMAGE_15` (Fal). Singleton `ImageService()`. | Requires `response.images[0].url` (Cloudflare URL), then product `register_cloudflare_url_asset`. Failures become HTTP 500 from `error.message`. |
| `mapgenerator/inpainting.py` | ACTIVE_RUNTIME | image edit / inpaint | `ImageService`, `ImageGenerationRequest`, `ImageModel`, `ImageSize` | `GPT_IMAGE_15`; mask + base image base64 | `images[0].url`. Raises on `success=False` or missing URL. |
| `routers/image_management_router.py` | ACTIVE_RUNTIME | image generation | `ImageService`, `ImageGenerationRequest`, `ImageSize` | Product `MODEL_MAP` → `FLUX_2_PRO` / `NANO_BANANA_PRO` / `GPT_IMAGE_15` | URL + product asset registry. Logs `metrics.duration_ms`, `model_used`, `retry_count`. |
| `shared/image_models.py` | ACTIVE_RUNTIME | image model enum adapter | `ImageModel` | Frontend IDs mapped onto GE enum | None (config only). |
| `cardgenerator/services/card_generation_service.py` images | ACTIVE_RUNTIME | image gen + image-to-image | `ImageService`, `MetricsService`, `ImageGenerationRequest`, `ImageModel`, `ImageSize` | `NANO_BANANA_PRO`; module-level `ImageService(metrics_service=...)` | `img_result.url`; logs duration/model/retries. |
| same file, `generate_item_description` | ACTIVE_RUNTIME (direct provider) | structured text **not** via GE | none (uses `openai.OpenAI().beta.chat.completions.parse`, `gpt-4o`) | Direct OpenAI | E3 candidate. Do not migrate in E2. |
| `playercharactergenerator/pcg_generator.py` | ACTIVE_RUNTIME | unstructured text | `generationengine.services.text_service.TextGenerationService`, `generationengine.models.requests.TextGenerationRequest`, `TextModel` | `GPT_5_1`; constructor `ValueError` → degraded health | Uses `success`, `content`, `metrics.tokens_used`. No `response_schema`. |
| `routers/ruleslawyer_router.py` | DORMANT imports | none via GE | unused `TextGenerationService`, `TextGenerationRequest`, `TextModel` (internal modules) | Runtime query path uses `AsyncOpenAI` + `generate_bot_response_stream` | Streaming is product-owned SSE, not `generate_stream`. |
| `tests/test_map_inpainting.py` | TEST_ONLY | image response types | `generationengine.models.image_responses.ImageGenerationResponse`, `ImageResult` | URL-shaped results | Compatibility for response models. |
| `tests/statblocks_v1/*`, `scripts/run_statblocks_v1_tests.sh` | HISTORICAL_DOC / isolation | none | comments listing `generationengine` as a dep tests should not need | N/A | Not a runtime consumer. |
| `pyproject.toml` | ACTIVE_RUNTIME dep | package | `generationengine = { git = "https://github.com/Drakosfire/GenerationEngine.git" }` | unpinned git default branch | Also declares `fal-client` and `openai` directly. |

### Not consumers

```text
DungeonMindBuddy     no generationengine imports
StoreGenerator       no generationengine imports
statblocks_v1 runtime no generationengine imports
generate_stream      no DungeonMindServer callers
IGenerator           no implementers in Server
OpenAIImageProvider  no Server imports
UploadService        not imported by Server (reached only via ImageService)
```

---

## Ownership direction already visible in MapSpec

MapSpec is the cleanest current split:

```text
DungeonMindServer owns MapSpec schema, map prompts, validation, HTTP behavior
GenerationEngine owns (today, imperfectly) OpenAI execution, schema-strict adaptation, retries
```

E2 should preserve that split and supply inference-call truth. It should not absorb `MapSpec`.

---

## Defects E2A records and does not fix

```text
direct AsyncOpenAI text execution
TextModel / pricing limited to gpt-5.1
Fal-only ImageService wiring despite OpenAIImageProvider
undeclared fal-client
mandatory Cloudflare uploader construction
product vocabulary on IGenerator.generator_type
SSE framing and [DONE]/[ERROR] sentinels inside the engine
prompt retention in GenerationMetrics.input
retry_count not observed from actual attempts
structured parse failure reported as success
tracked bytecode / no gitignore / no CI
pytest in runtime dependencies
ruff extra not installed by default sync
4 failing tests on clean baseline
README examples/claims drift (GPT_4O, optional Cloudflare, dual image providers)
```
