# E2 successor slices

## E2A — Characterize and define the contract (merged)

See `docs/CORE-CONTRACT.md` and `docs/CURRENT-STATE.md`.

## E2B — Trustworthy core primitives (this PR)

Landed: clean install/CI, `InferenceObservation`, `FailureCode`, catalog/profiles, `TextProvider`/`ImageProvider`, prompt-free legacy metrics, dead-API removal.

Active DungeonMindServer imports were not migrated. That is sequencing, not a compatibility promise.

## Next — Coordinated flag-day cutover

One controlled GenerationEngine + DungeonMindServer change:

```text
GenerationEngine
  move OpenAI behind TextProvider
  provider-neutral text / structured generation
  transport-neutral streaming (delete SSE core surface)
  truthful InferenceObservation population
  artifact-free image generation/editing
  truthful Fal/OpenAI wiring
  remove Cloudflare persistence from inference core

DungeonMindServer
  migrate every GE consumer
  own action -> generic profile mapping
  own Cloudflare / durable artifact persistence
  migrate appropriate direct provider calls

then immediately
  delete old GE facades
  delete legacy metrics/errors/responses
  delete stale TextModel/MODEL_PRICING
  delete UploadService from inference core
```

The running deployed instance is untouched until that development state is proven and deliberately deployed.

There is no E3 compatibility period for the old GenerationEngine contract.

## After cutover — settling

Clean-install matrices, fake-provider contract proofs, DungeonMindServer integration, documentation truth, Gate G.
