# GenerationEngine current state

**Branch:** `feat/e2-provider-neutral-cutover`  
**Target contract:** [CORE-CONTRACT.md](CORE-CONTRACT.md)  
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

Image publication, product prompts, schemas, and action→profile mapping belong to products.
