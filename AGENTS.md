# Agent operating policy

This file is durable repository law for agents working in GenerationEngine.

GenerationEngine owns provider-agnostic inference execution and inference-call truth. Products own prompts, schemas/domain meaning, workflows, authorization, fallback policy, and artifact persistence/publication.

## Ecosystem execution core — overmind-agent-core-v1

These rules are intentionally shared across active DungeonMind ecosystem repositories. Repository-specific law may add constraints, but it must not weaken this core.

1. **Re-anchor before action.** Fetch the current remote default branch and inspect relevant open PRs/active work before editing, reviewing, or merging. Chat history, stale handoffs, and local `main` are not current authority.
2. **Respect ownership boundaries.** Cross-repository architecture and sequencing belong in DungeonOverMind; runtime/product implementation belongs in the repository that owns the capability. When a change crosses owners, name the contract.
3. **Handoffs are portable bounded contracts.** A handoff may live on `main`, a branch, a PR, or another durable pinned ref/location. Its location alone neither activates nor invalidates it. Execution authority comes from explicit authorization/status, a pinned authority/ref, and bounded scope/write ownership. Do not require a handoff to be merged to `main` unless the specific workstream explicitly makes that a gate.
4. **Finish authorized implementation work all the way to a PR.** Once implementation is authorized, ordinary completion includes: implement → test/verify → inspect the cumulative diff → commit intended changes → push the branch → open or update the assigned PR. If no PR exists, open it. Do not stop with intended work only local, uncommitted, or unpushed and wait for another prompt to commit/push/open the PR.
5. **Merge is separate authority.** Opening/updating a PR is part of implementation completion; merging it is not. Merge only when the user or the repository's explicit process authorizes merge.
6. **Use isolated Git lanes.** Do not develop on local `main`. Use a branch/worktree or equivalent isolated checkout, and treat file/runtime/state collisions as coordination problems rather than relying on Git conflicts.
7. **Keep slices bounded.** One implementation slice should deliver one independently useful capability. A second capability, new durable/public contract, or unplanned extra PR is a stop/split signal unless explicitly authorized.
8. **Verify at the owning boundary.** Review the exact cumulative base→head diff and prove behavior at the layer that owns the invariant. A green helper test is not evidence for a boundary it does not exercise.
9. **Settle after merge.** Re-anchor, synchronize mutable authority that now became stale, and prune superseded process/transition scaffolding. Git history is the default archive; preserve a separate archive copy only when it carries unique durable evidence.

## Authority and pickup

Read the smallest current authority needed:

1. `README.md` — public usage and repository role.
2. `docs/README.md` — documentation authority/index.
3. `docs/CORE-CONTRACT.md` — stable semantic/public contract.
4. `docs/CURRENT-STATE.md` — implemented providers/capabilities/current limitations.
5. adopted decision docs such as `docs/STRUCTURED-CONFORMANCE.md`.
6. current code/tests and any explicitly authorized handoff.

Archive/transition handoffs are evidence only after their work settles.

## Ownership restrictions

- Keep provider/model execution, generic retries/deadlines, provider wire translation, structural conformance, normalized failures, lifecycle, and inference observations product-neutral.
- Do not move product prompts, campaign/statblock/rules concepts, product schemas, Agent-loop semantics, persistence, or product fallback policy into GenerationEngine to reduce imports.
- Caller-explicit target intent remains caller-owned; GenerationEngine executes and reports truthfully.
- Provider-specific adapters may contain provider-specific wire mechanics, but public contracts should expose reusable execution semantics rather than product/provider leakage.
- Experiments may use bounded provider-specific paths when the shared contract cannot express the experiment; do not quietly promote an experiment into the library contract.

## Engineering evidence

- Prefer focused contract/provider-free tests first, then the repository's broader CI gates when shared execution contracts change.
- Preserve truthful unknowns in usage/cost/latency telemetry rather than inventing zeros or guesses.
- Retry layers must remain explicit and bounded; provider SDK retries, transport retries, structural repair, and product retries are distinct concerns.
- Resource lifecycle is part of correctness. Concurrent failure/cancellation must drain owned work before closing shared provider resources.
- Update contract/current-state docs only when their claims actually change; completed migration handoffs belong in archive or Git history.
