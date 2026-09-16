# GenerationEngine documentation

**Status:** The E2 GenerationEngine cutover is implemented and accepted by Settling Gate G. `GenerationClient` owns live provider execution, images return bytes, and product publication is outside the inference core.

The next contract refinement is provider-independent structured conformance: products own schema definition/domain meaning; GenerationEngine owns making inference output conform structurally to the caller-supplied schema.

| Document | Purpose |
| --- | --- |
| [CURRENT-STATE.md](CURRENT-STATE.md) | Implemented package surface, current limitations, and execution behavior |
| [CORE-CONTRACT.md](CORE-CONTRACT.md) | Product-neutral inference contract established during E2 |
| [STRUCTURED-CONFORMANCE.md](STRUCTURED-CONFORMANCE.md) | Adopted refinement of structured generation: local schema validation, bounded corrective inference retries, and ownership boundary |
| [COMPATIBILITY.md](COMPATIBILITY.md) | Consumer inventory for the flag-day cutover, not an API-support promise |
| [E2-SUCCESSOR-SLICES.md](E2-SUCCESSOR-SLICES.md) | Historical E2B vs coordinated cutover sequencing |

When `CORE-CONTRACT.md` §8 is read, `STRUCTURED-CONFORMANCE.md` is the adopted refinement: provider-native strict-schema features are implementation strategies, not the semantic definition of `generate_structured()`.

Architecture authority for ecosystem ownership remains in DungeonOverMind.
