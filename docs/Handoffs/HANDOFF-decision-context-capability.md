# Request-bound decision context capability

Status: ACTIVE, PRIME design/implementation; serial GE foundation then dependent Buddy adoption. Base5c3c67cd5a088242f6a28421be0951e834cef2f8; branch codex/decision-context-capability; reuse clean owning checkout. No open GE PRs at activation. No service/database/state writes, new checkout or environment. Existing immutable runtime dependency remains untouched.

## Invariant and critique
Products choose context policy for actual prepared request bytes; GE owns serialization and the provider header. An optional selector on DecisionDispatchGate receives the immutable default prepared request and returns a closed DecisionContextMode enum. Only DEFAULT or LONG_CONTEXT is accepted (not strings, bools, arbitrary headers/options). GE freezes that selection into the final PreparedDecisionDispatch before authorization and lifecycle recording, then adds only its owned TypeSafe long-context header. Preserve exact body/model/endpoint/header/stream immutability and one wire attempt. Selector error or invalid result refuses before sending. No selector means standard context, not a compatibility execution path. No fallback, retry, truncation, prompt changes or product policy in GE.

Independent reviewer /root/bugbot_response_contract critiqued both progressive draft reads and this selector seam. Request-bound selection avoids duplicate serialization or unbound constructor flags and is the smaller owning-boundary change. Actual Gateway synthetic two-call contrast: identical body, default400 max_tokens_exceeded, long-context200 correct marker/input55337/output32. Evidence in PRIME visualization typesafe-long-context-gateway-probe; no retained story or APP writes. It proves header forwarding, not user answer quality or latency reliability.

## Exclusive expected writes
- This handoff.
- src/generationengine/decision_dispatch.py (closed mode, immutable final descriptor, optional selector callback).
- src/generationengine/providers/typesafe_dispatch.py (select/freeze/header/wire gate).
- tests/test_typesafe_dispatch.py (actual SDK/mock transport and mutation/refusal boundaries).
- docs/CORE-CONTRACT.md and docs/CURRENT-STATE.md (changed public contract only).
No other provider/public request/schema/client/model/lock/config files. Amend before expansion.

## Verification and delivery
Use actual SDK/mock transport tests: default header absent, selected long-context header present; final authorize/record see exact mode/body; malformed enum/selector exception fail known-not-sent; refusal sends zero; header/body/stream mutations rejected; concurrent requests independent; no retry on received rejection/unknown. Existing decision/provider suites and applicable CI. Inspect cumulative base→head diff, independent exact-head Bugbot review, commit/push/open PR; PRIME merge only satisfied gates. Buddy adoption separately binds mode into host admission/retained evidence, permits only needed phases, uses documented262k ceilings and retains default limits elsewhere. Original Conks Retry/answer/citations/state and authoring neighbors are downstream gates, not claimed by GE tests.
