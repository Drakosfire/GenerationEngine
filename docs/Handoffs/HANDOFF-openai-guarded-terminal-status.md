# Guarded OpenAI response terminal truth

ACTIVE serial repair for ARCHITECTURE HOLD on Buddy #1279 head
f4c095bc5618cc908d285ee83e4ac86f8488920d. PRIME implements solo; independent
ARCHITECTURE review required. Base 4aade6b34d64edbdc3d1cacb0a89d185663e2268;
branch codex/openai-guarded-terminal-status, non-serving
/tmp/prime-ge-authorized-dispatch. GE #21 is settled. No overlapping open GE PR.

One invariant: guarded nonstreaming Responses returns successful text only for
actual SDK status completed, no response error, and usable nonblank output text.
HTTP200/headers plus residual text do not prove success. Failed/cancelled/queued/
in_progress/incomplete/missing/unknown status, completed-with-error and empty output
must fail without retry. Preserve safe IDs/model/usage; raw residual body/errors
must not leak into public failure observations. Unguarded provider path unchanged.

Exclusive writes: this handoff; docs/Handoffs/HANDOFF-openai-responses-dispatch.md;
src/generationengine/providers/openai_text.py; tests/test_openai_dispatch.py;
docs/CORE-CONTRACT.md; docs/CURRENT-STATE.md; docs/README.md.
No new public schema, dependency, provider route, runtime, model calls or retained
writes. Actual SDK/mockHTTP failed-status residual-valid-text regression plus all
terminal variants; preserve header acknowledgment, one send, known usage/IDs,
noncompleted observation and no residual leakage. Run shared GE tests/lint/lock/build
and core-only wheel gate. Buddy separately pins accepted repair and reproduces
failed-status no-completion at its real APP boundary before #1279 may merge.
