# Vercel text provider

ACTIVE implementation lease: PRIME, branch `codex/vercel-text-provider`, beginning
at remote main `cd7287de4c4c53723a5530a949f1de05730f89da`.
Authority: DungeonOverMind#60, merge `e7c74bb00ff894825e507e42102098463d3b78b4`,
`Docs/Plans/HANDOFF-GE-ingestion-experiment-targets.md`.

Exclusive writes: new `src/generationengine/providers/vercel_text.py`, its tests,
lazy Vercel registration in `client.py` and `resolver.py`, the Vercel optional dependency/lock, and
public provider documentation. No other owner has an open overlapping GE PR.
Use the existing Buddy Python environment; do not create another environment.

Deliver explicit provider=`vercel` with arbitrary caller-selected model ID through
the documented fixed AI Gateway Chat Completions endpoint. API key is read by the
adapter. Preserve prompts, schema and provider identity; no model allowlist,
endpoint override, model substitution, product policy or new fallback. Native
schema output remains subject to existing GE local conformance. GE owns retry,
deadline, failures, usage, response IDs and resource cleanup. Unknown metrics stay
unknown. Stream deltas are provisional until a valid stop terminal; refusals,
truncation, missing terminals, tools and malformed choices fail closed.

Verify actual SDK/mock HTTP serialization, explicit IDs, credentials, omission and
forwarding of controls, local schema conformance, refusal/incomplete/malformed
terminal truth, usage, transport errors, cancellation/stream closure and lazy
registration. Inspect exact cumulative diff; independent Bugbot review and CI;
commit/push/PR and PRIME merge. No live provider calls or Buddy/benchmark changes
in this slice. Hand back exact merge revision to Buddy for its separate lease.

Official wire sources: Vercel AI Gateway documentation under
`/docs/ai-gateway/sdks-and-apis/openai-chat-completions`, especially
`chat-completions`, `structured-outputs` and `reasoning` (checked 2026-10-27).
