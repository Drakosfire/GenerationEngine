# GenerationEngine documentation

**Status:** The GenerationEngine half of the E2 cutover is implemented.
`GenerationClient` owns live OpenAI/Fal execution, images return bytes, and
product publication is outside the inference core. Ecosystem acceptance still
requires the paired consumer merge and Settling Gate G.

| Document | Purpose |
| --- | --- |
| [CURRENT-STATE.md](CURRENT-STATE.md) | Implemented package surface and execution behavior |
| [CORE-CONTRACT.md](CORE-CONTRACT.md) | Implemented product-neutral inference contract |
| [COMPATIBILITY.md](COMPATIBILITY.md) | Consumer inventory for the flag-day cutover, not an API-support promise |
| [E2-SUCCESSOR-SLICES.md](E2-SUCCESSOR-SLICES.md) | E2B vs coordinated cutover |

Architecture authority for ecosystem ownership remains in DungeonOverMind.
