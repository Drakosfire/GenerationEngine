# GenerationEngine current state

**Branch:** `feat/e2-provider-neutral-cutover`  
**Contract:** [CORE-CONTRACT.md](CORE-CONTRACT.md)  
**Structured-conformance refinement:** [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md)  
**Consumer inventory:** [COMPATIBILITY.md](COMPATIBILITY.md)

```text
public API: GenerationClient
  generate_text
  generate_structured
  stream_text
  generate_image
  edit_image
live adapters: OpenAITextProvider, FalProvider
observations: InferenceObservation on success and failure
failures: FailureCode / GenerationEngineError (no SDK types)
image results: bytes only; no Cloudflare, no URLs
catalog: generic profiles + live catalog ids required by DungeonMindServer
deleted: TextGenerationService, ImageService, UploadService, MetricsService,
         TextModel, ImageModel, MODEL_PRICING, generationengine.models,
         generationengine.services
```

Image publication, product prompts, schema definitions/domain meaning, and action→profile mapping belong to products.

## Structured generation status

`generate_structured()` exists today, but `main` does **not yet** implement the full provider-independent structured-conformance contract adopted on 2026-09-16.

Current provider adapters still carry provider-native structured-output mechanics. The adopted target is broader:

```text
caller supplies JSON Schema
→ GenerationEngine chooses provider-specific structured strategy
→ GenerationEngine parses locally
→ GenerationEngine validates locally against the schema
→ GenerationEngine performs bounded corrective inference retries when needed
→ return schema-conforming parsed object OR STRUCTURED_OUTPUT_INVALID
```

Provider-native strict schema, JSON-object modes, and prompt/instruction steering are implementation strategies, not the public semantic definition of `generate_structured()`.

Products retain domain/business/evidence validation. GenerationEngine owns only structural conformance required to fulfill the inference request.

Until [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md) is implemented and accepted, consumers may still contain structural parse/repair loops and bounded lab-specific provider paths. Those are migration sources, not the desired steady state.
