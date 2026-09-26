# HANDOFF — GEJ-02 TypeSafe/Jev decision provider

**Status:** DEFERRED DRAFT — stacked on GEJ-01  
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

At activation, rebase this branch onto the accepted GEJ-01 merge and refresh current TypeSafe SDK/Gateway behavior.

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
