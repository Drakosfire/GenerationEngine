# GenerationEngine current state

**Branch:** `e5/structured-conformance`  
**Contract:** [CORE-CONTRACT.md](CORE-CONTRACT.md)  
**Structured-conformance refinement:** [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md)  
**Consumer inventory:** [COMPATIBILITY.md](COMPATIBILITY.md)

```text
public API: GenerationClient
  generate_text
  generate_structured   # GE local schema validation + at most one structural repair
  stream_text
  generate_image
  edit_image
text dispatch: openai, openrouter (by provider identity)
openrouter structured: JSON instructions, not native json_schema
live adapters: OpenAITextProvider, OpenRouterTextProvider, FalProvider
observations: InferenceObservation on success and failure
  retry_count = transport retries
  conformance_retry_count = structured repairs
  provider_attempt_count = all provider generate calls
failures: FailureCode / GenerationEngineError (no SDK types)
image results: bytes only; no Cloudflare, no URLs
catalog: selection/metadata authority for profile defaults and governed model IDs
explicit targets: provider + model may execute without a LIVE_MODELS row
deleted: TextGenerationService, ImageService, UploadService, MetricsService,
         TextModel, ImageModel, MODEL_PRICING, generationengine.models,
         generationengine.services
```

Image publication, product prompts, schema definitions/domain meaning, and action→profile mapping belong to products.

Labs should prefer GenerationEngine explicit targets when the experiment fits the generic contract. A bounded lab-only direct provider path remains allowed when GenerationEngine cannot yet express a required control; that path must not become a production seam.

## Structured generation status

`generate_structured()` implements the provider-independent structured-conformance contract in [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md).

```text
caller supplies JSON Schema
→ GenerationEngine chooses provider-specific structured strategy
→ GenerationEngine parses locally
→ GenerationEngine validates locally against the schema
→ GenerationEngine performs at most one corrective inference retry when needed
→ return schema-conforming parsed object OR STRUCTURED_OUTPUT_INVALID
```

Provider-native strict schema, JSON-object modes, and prompt/instruction steering are implementation strategies, not the public semantic definition of `generate_structured()`.

Products retain domain/business/evidence validation. GenerationEngine owns only structural conformance required to fulfill the inference request.

Consumers may still contain domain/evidence validation and, until migrated, leftover structural parse/repair loops. Those loops are migration sources, not the desired steady state.
