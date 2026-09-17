# GenerationEngine current state

**Branch:** `e5/explicit-target-provider-dispatch`  
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
text dispatch: openai, openrouter (by provider identity)
openrouter: generate_text + stream_text only
generate_structured via OpenRouter: UNSUPPORTED_CAPABILITY
  (provider-native json_schema is not the structured contract)
live adapters: OpenAITextProvider, OpenRouterTextProvider, FalProvider
observations: InferenceObservation on success and failure
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

`generate_structured()` exists today, but this tree does **not yet** implement the full provider-independent structured-conformance contract adopted on 2026-09-16. See [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md).

OpenAI may still submit provider-native JSON Schema as a provider-specific optimization. OpenRouter does not: `generate_structured()` through OpenRouter fails closed with `UNSUPPORTED_CAPABILITY` and does not send `response_format=json_schema`. Labs that need OpenRouter/`json_object` plus local validation should keep that direct path until the conformance layer exists.

The adopted target is broader than any provider-native schema feature:

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
