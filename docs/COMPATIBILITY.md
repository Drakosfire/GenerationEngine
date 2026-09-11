# Cutover inventory

GenerationEngine is a single-owner system. There is no long-lived compatibility architecture for the demolished facades.

```text
deleted public API
  TextGenerationService
  ImageService
  MetricsService
  UploadService
  TextGenerationRequest / ImageGenerationRequest
  TextModel / ImageModel / ImageSize
  generationengine.models.*
  generationengine.services.*

current public API
  GenerationClient
  TextRequest / TextResult
  ImageRequest / ImageResult / GeneratedImage
  GenerationEngineError
  InferenceProfile / InferenceObservation / FailureCode
```

DungeonMindServer must consume only the current public API. Do not rebuild adapters, deprecation layers, dual APIs, SSE adapters, or URL-returning shims for the deleted surfaces.
