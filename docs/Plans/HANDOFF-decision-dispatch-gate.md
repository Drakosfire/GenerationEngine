# Decision dispatch authorization boundary

ACTIVE. PRIME implementation solo; ARCHITECTURE independent review. Operator goal
requires ordinary Buddy Jev acquisition plus one writer under durable authorization.
GenerationEngine owns provider wire/lifecycle truth; Buddy owns authorization policy,
roles, source scope, persistence and Graph broker. No runtime reads OverMind config.

Baseafe824d1e54cf7d54a58bc9aa92a1cf0234f73ec; branch
codex/decision-dispatch-port; isolated /tmp/prime-ge-authorized-dispatch.
Serial prerequisite before Buddy consumer; no open GE PRs/overlapping leases found.

One capability: opt-in prepared-decision gate at official TypeSafe SDK's supported
injected transport. Expose immutable actual serialized body bytes/hash, provider,
model, method and exact URL without secret headers. Guarded requests disable SDK
compression so approved bytes are the wire entity; SDK/transport retries and
redirects disabled. Product authorizes before I/O and acknowledges lifecycle records.
Deny/callback failure before send guarantees not_sent; after send begins ambiguous
failure/cancellation is outcome_unknown, response headers acknowledge response_received.
Parse failure does not erase acknowledged HTTP response. Guarded failures must never
trigger GE retries. Default provider behavior unchanged.

Writes: this handoff; src/generationengine/decision_dispatch.py;
src/generationengine/providers/typesafe_dispatch.py;
src/generationengine/providers/typesafe_decision.py; tests/test_typesafe_dispatch.py;
docs/CORE-CONTRACT.md; docs/CURRENT-STATE.md; docs/README.md.
No product prompts/roles/persistence, credential changes, provider/model selection,
new provider, live calls, SDK monkeypatch or serving changes.

Verify actual installed SDK serializer+mock HTTP transport, Unicode/escaping and all
question types; exact approved/sent bytes; endpoint/model/body/header mutation denial;
authorization/callback cancellation/error before send; timeout/cancel after send;
HTTP response+parse failure; one attempt even when caller permits retries; resource
closure and concurrency. Independently review exact cumulative diff and broader GE
gates for the shared optional contract before merge. Buddy adoption remains separate.
