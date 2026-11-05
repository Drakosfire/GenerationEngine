# INFERENCE: retain provider-reported charges

Status: ACTIVE. User explicitly authorized token/price tracking and merge. Root owns this bounded implementation; independent subagent review. Cross-owner contract: GenerationEngine owns inference observations/attempt truth; Buddy owns retention and per-run display. This adopts the existing cost-accounting ownership in CORE-CONTRACT, not a product workflow in GE.

Serial delivery: this GE PR first, then Buddy consumer pin/UI/retention PR. GE branch codex/provider-charge-accounting, basee9366efad89b76ad0a62b1efacb0d8353120af0d, isolated clone /tmp/ge-charge-accounting (managed worktree tool cannot allocate this additional repo). No other open GE PRs at activation. Shared product interpreter/services are excluded; test environment /tmp/ge-charge-venv.

Scope: OpenRouter usage.cost→provider TextGenerationResult→single-attempt callback→whole operation and stream observation. Actual provider USD only; no guessed rates. Unknown and partial charged usage remain distinct from reported complete totals; zero is valid; cached/reasoning tokens remain subsets. No provider/model execution required for validation. User authorizes code merges, not product data publication.

Write lease: src/generationengine/providers/base.py,errors.py,openrouter_text.py; src/generationengine/client.py,types.py,observation.py; tests/test_provider_charges.py; this handoff; docs/CURRENT-STATE.md and CORE-CONTRACT.md cost semantics only.

Failure critique: missing/invalid cost cannot become zero or break valid token responses; retries/invalid conformance attempts cost money; callback and aggregate receipts must not double-count; partial streams remain incomplete; reported charge source differs from catalogue rate provenance. No SDK retries or pricing network lookup is added. Validate at adapter/client/callback boundaries, unknown/subtotal aggregate paths, stream success/error, and broader offline suite. Inspect cumulative diff, review, commit/push/PR before merge. Product historical receipts remain unchanged/unavailable unless they contain actual charge evidence.

Validation:432 provider-free tests passed in dedicated /tmp/ge-charge-venv, including wheel build/import checks and new charge cases. Ruff/diff checks pass. No provider calls, shared package installation, services or product databases touched. The initial test environment lacked pytest-asyncio/writable UV cache; failures retained and rerun in dedicated environment.
