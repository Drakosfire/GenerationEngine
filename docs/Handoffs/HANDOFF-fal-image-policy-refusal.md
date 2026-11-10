# Fal image refusal normalization

ACTIVE PRIME serial owner repair from563ae839cd9fef9510e0e389e115ebb8671eb8ed, branch codex/fal-image-policy-refusal in the existing GenerationEngine checkout. No open GE PR or active caller edits overlap the four allowlisted paths below. Qualification uses the owning checkout’s standard .venv with locked dependencies; the previously reused environment has an active runtime consumer and is excluded. No new checkout or per-slice source export. PRIME owns the qualification environment; retain it for subsequent GE verification and do not use it as a product runtime.

A retained fal image request returned422 with structured detail.type=content_policy_violation. The real SDK wraps its httpx HTTPStatusError, so the adapter's generic exception branch currently converts a known non-retryable refusal into PROVIDER_ERROR. The owning CORE-CONTRACT already defines PROVIDER_REFUSED for this condition.

Outcome: recognize only that explicit422 structured refusal on the SDK's preserved HTTP cause, return a fixed safe provider-neutral refusal, and preserve the original exception chain. Other SDK errors retain existing generic unknown classification; no message guessing, automatic retry, new code, product knowledge or policy bypass.

Exclusive writes: src/generationengine/providers/fal_provider.py; tests/test_fal_provider.py; docs/CURRENT-STATE.md; this handoff.

Validate the actual SDK exception shape through GenerationClient generation/editing, non-retryable refusal/state and no raw provider/prompt leakage. Contrast unknown422, malformed payloads and nonmatching statuses. Run provider-free owner tests, full local GE tests, lint and lock/build gates with available shared dependencies. Independent exact-head review, commit/push/PR, then PRIME merge.

Dependency order: this independently useful GE fix first; a separate small Buddy PR adopts the merged GE pin and translates existing normalized refusal into an actionable product message. Buddy must not parse Fal SDK exceptions or infer refusal from prose. Existing original failure stays private and retained. No new provider call or fresh story transfer is authorized by this handoff.
