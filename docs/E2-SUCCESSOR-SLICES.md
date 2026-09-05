# E2 successor slices

## E2A — Characterize and define the contract (merged)

See `docs/CORE-CONTRACT.md`.

## E2B — Trustworthy core primitives

Catalog, observations, failure taxonomy, provider protocols, packaging/CI.

## E2 — Coordinated flag-day cutover (this PR)

`GenerationClient` is the live execution surface. OpenAI and Fal run behind adapters. Image results are bytes. Legacy facades are deleted.

Paired DungeonMindServer work migrates every inventoried consumer in the same cutover unit.

## After cutover — Settling Gate G

Clean-install matrices, fake-provider contract proofs, DungeonMindServer integration, documentation truth. Gate G is not part of this PR.
