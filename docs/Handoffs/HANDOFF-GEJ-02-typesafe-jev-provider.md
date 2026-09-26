# HANDOFF — GEJ-02 TypeSafe/Jev decision provider

**Status:** READY FOR RE-REVIEW — review blocker fixed; do not merge before review
**Repository:** `Drakosfire/GenerationEngine`
**Physical base:** `gej/01-generic-decision-capability`
**Cross-repo authority:** `Drakosfire/DungeonOverMind/Docs/Plans/STACK-rules-ingestion-generationengine-sidequest.md`
**Primary question:** Can GenerationEngine execute the generic decision contract through the official TypeSafe client and current Jev Gateway route while preserving truthful inference-call observations and normalized failures?
**Predecessor:** `GEJ_01_GENERIC_DECISION_CAPABILITY_ACCEPTED`
**Unlocks:** RIGE-01

## Stack position

```text
GEJ-01 generic decision capability
  ↓
GEJ-02  ← YOU ARE HERE
  ↓
RIGE-01 semantic-lifting migration
  ↓
RIGE-02 retrieval/eval migration
  ↓
RIGE-03 ingestion-time migration
  ↓
RIGE-04 provider demolition
  ↓
RLH-05 resumed review
```

Activation re-anchor (2026-09-26): GEJ-01 is accepted at PR #15 head `11acfa8dddb4366e85933d40f7c3801c24f6451f` but remains unmerged by user instruction. This shell was rebased from its planted predecessor `27ed6cb7a889579b968de8f057a4e34b5ba7ae1f` onto GEJ-01, then restacked onto its Choice-description amendment after RIGE-01 exposed the need to preserve label meaning. GenerationEngine `main` remains `19cf68dceb5f4ec7a3d20e17ae6d5c9c8d7aeaa5`; PR #14 changes governance only and has no implementation-path collision. The official `typesafe-sdk` 0.7.1 has `AsyncTypeSafeClient`, `Noul`, `Choice`, `Score`, and `RetryPolicy(max_retries=0)`. The existing RulesIngestion Jev pilot confirms the current Gateway route and response shapes. This lease includes `uv.lock`, focused provider-contract tests, and an opt-in smoke script required to make the live witness executable. Because CI runs the full test suite with explicit extras, `.github/workflows/ci.yml` must include the new optional extra in its test environment; the provider-free wheel import remains separate.

## Provider identity

Implement a provider adapter for the generic decision surface.

Target current route:

```text
GenerationClient.decide
→ DecisionProvider
→ TypeSafeDecisionProvider
→ official typesafe-sdk
→ https://ai-gateway.vercel.sh/typesafe
→ requested model typesafe-ai/jev
```

Use the project credential:

```text
TYPESAFE_JEV_API_KEY
```

GenerationEngine owns reading this credential. Never expose it to consumers, observations, logs, fixtures, or exceptions.

The upstream SDK may default to another environment name; pass the project key explicitly rather than mutating process environment.

## Public/provider vocabulary mapping

Provider-local mapping may use:

- GE Binary decision ↔ TypeSafe Noul;
- GE Choice decision ↔ TypeSafe Choice;
- GE Score decision ↔ TypeSafe Score.

Those TypeSafe names stay inside the adapter.

Support optional binary true/false descriptions if the current SDK supports them.

## Async requirement

GenerationEngine's public client is async. If the official TypeSafe SDK call is synchronous at implementation time, do not block the event loop. Use the narrowest safe async bridge (for example `asyncio.to_thread`) or the SDK's official async client if one exists.

GE owns the overall deadline/retry loop; disable or account for SDK retries so attempts are not hidden.

## Model and transport truth

For the current Gateway route, a response may report only `typesafe-ai/jev`.

Record truthfully:

```text
provider = typesafe
provider_transport = vercel_ai_gateway
requested_model = caller model
resolved_model = GE target
response_model = provider-reported value
```

Do not claim a concrete upstream Jev version unless the provider actually returns one.

## Failure mapping

Map provider/SDK failures into existing GE codes:

- missing key/extra → `CONFIGURATION_UNAVAILABLE`;
- malformed generic request before SDK → `INVALID_REQUEST`;
- auth/permission → a stable existing provider/config failure chosen consistently with current contract;
- rate limit → `RATE_LIMITED`;
- timeout → `PROVIDER_TIMEOUT`;
- transport/5xx → `PROVIDER_UNAVAILABLE` when appropriate;
- malformed typed response → `MALFORMED_PROVIDER_RESPONSE`;
- unknown SDK failure → `PROVIDER_ERROR`.

No SDK exception crosses the public surface.

## Dependency packaging

Add a dedicated optional extra, e.g.:

```toml
typesafe = ["typesafe-sdk>=0.7.1,<0.8"]
```

Do not add TypeSafe to GenerationEngine's base dependency set.

Update installation/environment docs accordingly.

## Suggested lease

```text
src/generationengine/providers/typesafe_decision.py
src/generationengine/client.py                 # lazy provider wiring only
src/generationengine/providers/errors.py       # only if mapping requires
pyproject.toml
README.md
docs/CORE-CONTRACT.md
docs/CURRENT-STATE.md
tests/test_typesafe_decision.py
tests/test_client.py / test_explicit_targets.py as needed
docs/Handoffs/HANDOFF-GEJ-02-typesafe-jev-provider.md
```

## Live witness

Add an opt-in real-provider smoke proof at the GenerationEngine boundary.

It must demonstrate all three public question families and expose:

- provider;
- transport;
- requested/resolved/response model identities;
- usage if the provider supplies it;
- typed answers;
- no secret material.

Do not make live TypeSafe access a normal CI requirement.

## Acceptance token

```text
GEJ_02_TYPESAFE_JEV_PROVIDER_ACCEPTED
```

## Stop conditions

Stop instead of widening if:
- the current SDK cannot preserve the generic decision shapes;
- the only implementation would block the async event loop;
- retries cannot be made observable/bounded;
- Gateway routing requires product code or browser credentials;
- accurate model/transport truth cannot be represented by the GEJ-01 observation contract.

## Implementation and acceptance evidence (2026-09-26)

- Exact intended PR base: GEJ-01 `11acfa8dddb4366e85933d40f7c3801c24f6451f`; tested implementation head after preserving Choice descriptions: `9ed73ad6e4219f649d2462ab4106c6f866d2ff85`. This evidence update follows as a separate commit.
- The official async SDK handles the three public question families through provider-local mapping. Its internal retries are disabled. GE owns the optional SDK dependency, project credential, Gateway route, failure mapping, deadline, retries, and observation fields. No provider vocabulary entered the generic public decision contract.
- Focused tests: 14 passed. Full Python 3.11 and 3.13 CI test matrices: 182 passed each, with `uv lock --check`, Ruff, and `uv build` passing. After restacking onto the Choice-description amendment, the Python 3.11 full suite and Ruff passed again (182 tests). The isolated provider-free wheel import test passed within the full suite; CI now installs the optional SDK extra for provider tests while still checking the base wheel separately.
- Opt-in live Gateway smoke passed with Binary, Choice, and Score answers, `provider=typesafe`, `provider_transport=vercel_ai_gateway`, requested/resolved/response model each `typesafe-ai/jev`, and provider-reported usage of 351 input / 65 output tokens. The response did not identify a concrete upstream model version. No credential was emitted.
- Lease adjustments identified at activation: `uv.lock`, `scripts/smoke_typesafe_decision.py`, the CI extra, and the optional-dependency invariant test. The cumulative diff was reviewed against the exact GEJ-01 base and `git diff --check` passed.
- Limitation: the current TypeSafe API accepts text, object, or array state. This adapter rejects other JSON scalar states as `INVALID_REQUEST`; the provider-neutral GE decision contract remains broader.

The evidence above records the prior reviewed head. Its acceptance claim was withdrawn after review comment `5848291530` identified lost observation metadata on locally malformed answers.

## Review correction and re-review evidence (2026-09-26)

- Exact corrected predecessor and PR base: GEJ-01 `87d49c20c84983267dd7044b350e68806e10383c`. This branch was cleanly restacked from the prior GEJ-01 head `11acfa8dddb4366e85933d40f7c3801c24f6451f`; `git merge-base` confirms the corrected predecessor.
- Tested implementation head: `c5aaea2558d615adb5acebad684dba499015e70c`. The TypeSafe adapter now captures only validated response model, request ID, token usage, and Gateway transport before answer normalization. A rejected Score legend preserves those fields through the public `GenerationEngineError.observation` without exposing the malformed legend or response body.
- Focused decision and TypeSafe tests: 15 passed. Full Python 3.11 and 3.13 suites: 183 passed each. On each version, `uv lock --check`, locked sync with the CI extras, Ruff, and `uv build` passed. The provider-free wheel import remains covered by the full suite.
- Live Gateway smoke passed again with Binary, Choice, and Score answers; `provider=typesafe`, `provider_transport=vercel_ai_gateway`, requested/resolved/response model `typesafe-ai/jev`, 351 input tokens and 65 output tokens. No credential was emitted.
- Cumulative diff against the exact corrected GEJ-01 base contains only GEJ-02's provider adapter, optional package/CI wiring, docs/handoff, smoke script, and tests. `git diff --check` passed.

Acceptance token `GEJ_02_TYPESAFE_JEV_PROVIDER_ACCEPTED` is pending re-review of this corrected head.

## Optional usage correction and re-review evidence (2026-09-26)

- Review comment `5848629392` identified a second provider-contract issue: the SDK permits unknown input/output token counts. The prior adapter incorrectly classified those valid responses as malformed.
- Tested implementation head: `e38a6e7f1361a73e1520dacc7b130a42521ee120`; exact intended base remains accepted GEJ-01 `87d49c20c84983267dd7044b350e68806e10383c`. The adapter still requires a valid response model, now passes unknown usage counts as `None`, and rejects negative or incorrectly typed counts.
- A real `SystemOneResponse` validated with both token counts unknown now succeeds through `GenerationClient.decide` and retains unknown observation usage. Parameterized tests show negative, string, and boolean counts fail closed as `MALFORMED_PROVIDER_RESPONSE`.
- Focused TypeSafe and decision tests: 20 passed. Full Python 3.11 and 3.13 CI-equivalent suites: 188 passed each; `uv lock --check`, locked sync with CI extras, Ruff, and `uv build` passed on both. The provider-free wheel import remains in the full suite.
- Cumulative diff remains limited to GEJ-02's adapter, optional package/CI wiring, docs/handoff, smoke script, and tests; `git diff --check` passed. The previous live Gateway witness remains valid because this correction concerns absent usage fields and changes no request mapping.

Acceptance token `GEJ_02_TYPESAFE_JEV_PROVIDER_ACCEPTED` remains pending exact-head re-review. Do not merge before review.
