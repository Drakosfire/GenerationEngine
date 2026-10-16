# OpenAI Responses dispatch authorization

ACTIVE. PRIME implements; ARCHITECTURE independently reviews. Base
dde768f3e632af20b969cc694d8318265ce6ff5c; branch codex/openai-responses-dispatch,
isolated non-serving /tmp/prime-ge-authorized-dispatch. No overlapping GE PRs.

One capability: opt-in authorization/lifecycle over actual official OpenAI SDK
serialized nonstreaming Responses requests. GE owns wire truth; consumer owns
prompt, evidence, permissions, single-writer policy, storage and completion.
Expose immutable body/hash/size, attempt identity, provider, API mode, model,
method and exact destination without credential headers. Use supported SDK
http_client injection; no SDK internals, Agent runtime or monkeypatch.

Guarded dispatch permits only POST https://api.openai.com/v1/responses,
nonstreaming ordinary text, with tools=[] explicitly. Structural repair is
rejected before I/O; ordinary JSON-object mode has no repair. Disable SDK/HTTP/GE retries and redirects;
callback denial or failure prevents send. Record sending before transport entry,
known_not_sent before entry failures, outcome_unknown after entry failures, and
response_received on headers before SDK parsing. Consumers reconcile uncertain
or unacknowledged outcomes. Stream requests on guarded provider fail before I/O.
Injected SDK clients cannot bypass the gate. Default behavior stays unchanged.

Exact writes: this handoff; predecessor docs/Handoffs/HANDOFF-decision-dispatch-gate.md;
src/generationengine/text_dispatch.py;
src/generationengine/providers/openai_dispatch.py;
src/generationengine/providers/openai_text.py;
tests/test_openai_dispatch.py; docs/CORE-CONTRACT.md; docs/CURRENT-STATE.md;
docs/README.md. No dependency, schema, registry, runtime, corpus or retained state
changes; no live calls. Independent review of cumulative diff required.

Verify actual SDK mock-HTTP bytes (Unicode/instructions/options), authorization,
mutation, cancellation and callback failures, timeout, HTTP errors and malformed
responses; no retry when caller allows retries, distinct concurrent attempts,
resource closure, guarded-stream refusal and unchanged default tests. Broader GE
CI includes lint, lock check, tests and core-only wheel import. Buddy adoption
follows separately and does not become ready from this library prerequisite.
