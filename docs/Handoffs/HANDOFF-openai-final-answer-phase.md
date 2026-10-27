# OpenAI terminal answer phase selection

Status: ACTIVE. Owner: PRIME. Topology: serial, one GE implementation PR.
Base: origin/main 3e08516. Existing non-serving nested GE checkout, branch
codex/openai-final-answer-phase. No new worktree or environment.
Authority: operator instruction to own ADVERSARY-confirmed runtime regressions.

## Invariant and critique

Captured Buddy Q10 HTTP200/completed output contains commentary plus a separate
final_answer. SDK output_text concatenates both, causing structured parse failure.
The protected original request/response hashes and private replay remain with
ADVERSARY; commit synthetic shapes only. Official Responses phase documentation
labels commentary intermediate and final_answer completed output:
https://developers.openai.com/api/docs/guides/reasoning#phase-parameter

Independent Bugbot invariant critique approved exactly one final assistant message
selection at the OpenAI adapter. A single standard unphased message is valid only
when no phased/other answer creates ambiguity. Commentary-only/missing final is
incomplete; duplicate finals, unknown phase, mixed phased/unphased answers and
malformed content fail closed. Refusal precedes extraction. Never repair text,
choose by JSON validity, concatenate commentary, resend, or use aggregate fallback.
Keep actual status, IDs, model, usage and original wire/dispatch evidence.

Same selector owns ordinary, guarded and stream-terminal result interpretation.
Deltas remain provisional; they cannot authorize terminal success. No public
phase event/schema, model/prompt change, product evidence policy or raw history
migration. Product grounding and answer validation remain required afterward.

## Exclusive writes and proof

- This handoff.
- src/generationengine/providers/openai_text.py
- tests/test_openai_text.py
- tests/test_openai_dispatch.py
- docs/CORE-CONTRACT.md and docs/CURRENT-STATE.md for the changed terminal contract.

Red actual SDK/mockHTTP commentary+final regression; cover null phase, multiple
parts in one answer, missing/duplicate/unknown/mixed phases, refusal, invalid status,
stream terminal and retained IDs/usage, exactly one send/no retries. Preserve private
original failure. Run full provider-free tests/lint/lock/build and wheel import
using established dependencies. Independent exact cumulative-head Bugbot review,
commit/push/PR, PRIME merge. Buddy adoption of immutable reviewed dependency and
unchanged independent Q10 rerun follow separately; do not mutate its active frozen
benchmark source or installed environment. No provider calls in implementation.
