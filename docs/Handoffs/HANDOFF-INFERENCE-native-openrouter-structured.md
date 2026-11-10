# INFERENCE: native OpenRouter structured output

Status: ACTIVE. User authorization: fix GE-native structured output with no
prompt-only fallback, then rerun retained nine-model Q15 tasting excluding
unsupported routes. Product context, Gold and tasting remain Buddy-owned.

Serial topology: this GE PR first; Buddy PR2329 adopts its pinned GE revision.
Base: 18ea21dc2fbae4e55128dd18790676c890a78985.
Checkout: /media/drakosfire/Projects/DungeonOverMind/GenerationEngine/GenerationEngine
Branch: codex/native-structured-openrouter. Existing clean completed checkout reused.
Exclusive writes: this handoff, src/generationengine/providers/openrouter_text.py,
tests/test_openrouter_text.py, docs/CORE-CONTRACT.md, docs/CURRENT-STATE.md,
docs/STRUCTURED-CONFORMANCE.md. No services, databases or paid provider calls here.

Invariant: structured OpenRouter requests send strict native JSON Schema and
require parameter-supporting endpoints. Caller system/user text stays unchanged;
no schema guide is appended and no error-driven downgrade to prompt-only mode.
Keep local original-schema validation and explicit retry accounting. Preserve
ordinary text, explicit JSON-object mode, streaming and caller route restrictions.
Native schema projection reuses existing GE strict-schema owner.

Verify actual SDK bytes using mock transport, negative unsupported routes, local
validation despite provider success, all owner tests and broader local suite.
No GitHub CI. Preserve old tasting artifacts/environment; qualify a separate
candidate runtime only after owner fix is pinned.
