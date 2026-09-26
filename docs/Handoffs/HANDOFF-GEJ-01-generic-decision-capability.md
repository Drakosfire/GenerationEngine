# HANDOFF — GEJ-01 Generic typed-decision capability

**Status:** DEFERRED DRAFT — side quest shell, no implementation yet  
**Repository:** `Drakosfire/GenerationEngine`  
**Creation anchor:** `main@19cf68dceb5f4ec7a3d20e17ae6d5c9c8d7aeaa5`  
**Cross-repo authority:** `Drakosfire/DungeonOverMind/Docs/Plans/STACK-rules-ingestion-generationengine-sidequest.md`  
**Primary question:** Can GenerationEngine represent typed probabilistic decision inference as a generic capability without knowing Jev, rules, or RulesIngestion semantics?  
**Predecessor:** side-quest activation only  
**Unlocks:** GEJ-02

## Stack position

```text
GEJ-01  ← YOU ARE HERE
  ↓
GEJ-02 TypeSafe/Jev provider
  ↓
RIGE-01 RulesIngestion semantic-lifting migration
  ↓
RIGE-02 retrieval/eval migration
  ↓
RIGE-03 ingestion-time migration
  ↓
RIGE-04 direct-provider demolition
  ↓
RLH-05 rebase + resumed review
```

GenerationEngine PR #14 (governance) was open when this shell was planted. Re-census it and all active PRs before coding; if it touches only governance/docs, rebase rather than absorbing unrelated work.

## Ownership decision

Add a **generic decision capability**, not a Jev-shaped text-generation mode.

Public vocabulary should be provider-neutral:

```text
Capability.DECISION

BinaryDecisionQuestion
ChoiceDecisionQuestion
ScoreDecisionQuestion
DecisionQuestion union

DecisionRequest
  state                 JSON-safe value
  questions             named questions
  provider/model/profile selection
  deadline/retry controls

DecisionAnswer union
DecisionResult
  answers
  observation

DecisionProvider
GenerationClient.decide(...)
```

Naming may vary if current conventions strongly suggest better names. Public GenerationEngine types must not expose `Noul`, `Jev`, EvidenceUnits, rules terminology, or product-specific dispositions.

Initial execution may require explicit `provider + model`; **do not add a generic decision profile merely to make the first provider selectable**. A profile should be added only when there is reusable requirement-shaped selection pressure.

## Semantics

### Binary decision

Represents a yes/no judgment with probability/confidence information when the provider supplies it. The public contract may support optional human-readable descriptions for true/false meanings.

### Choice decision

Represents exactly one selected label from a named option set plus probability/confidence information when available.

Option order/identity must be deterministic at the public boundary. Products own what the labels mean.

### Score decision

Represents an ordered rubric with at least two levels and a provider result that can preserve continuous score plus distribution/confidence when available.

Products own interpretation of the scale.

## Observation refinement

RLH-01 exposed a model-attribution problem: a gateway may return only the same floating alias the caller requested.

GenerationEngine must distinguish:

```text
requested_model   # caller request
resolved_model    # GE resolution result / explicit target
response_model    # provider-reported model identity, possibly still an alias
provider_transport # optional execution transport, e.g. vercel_ai_gateway
```

Add the smallest generic optional observation field needed to identify an intermediary transport. Do not invent an upstream concrete model version when it is unavailable.

Unknown remains `None`; an alias remains an alias.

## Provider boundary

Add a capability-focused `DecisionProvider` protocol. Do not force decision semantics through `TextProvider.generate()`.

Decision provider SDK types must remain behind the provider adapter boundary exactly like text/image SDK types.

## Failure/retry behavior

Reuse existing GenerationEngine public failure vocabulary and overall deadline/transport-retry semantics where they apply.

GEJ-01 must define before GEJ-02 implements:

- invalid question/request → `INVALID_REQUEST`;
- missing provider capability/config → existing configuration/capability failures;
- provider rate limit/timeout/unavailable/malformed response → existing normalized failures;
- one GE decision operation → one `InferenceObservation`;
- provider answers are result payload, not observation fields.

No automatic generative-text fallback is part of this capability.

## Suggested lease

```text
src/generationengine/catalog.py
src/generationengine/types.py
src/generationengine/providers/base.py
src/generationengine/client.py
src/generationengine/observation.py
src/generationengine/__init__.py
tests/test_decision_*.py
tests/test_public_surface.py
docs/CORE-CONTRACT.md
docs/CURRENT-STATE.md
docs/Handoffs/HANDOFF-GEJ-01-generic-decision-capability.md
```

Touch resolver/catalog implementation only as required to support `Capability.DECISION` and explicit provider+model resolution.

## Do not

- add TypeSafe SDK code;
- add Vercel endpoint logic;
- add product-specific prompts/questions;
- add decision persistence or replay receipts;
- add Rules Lawyer/RulesIngestion vocabulary;
- convert ordinary structured text into decisions;
- change existing text/image behavior.

## Acceptance proof

Focused tests must prove:

1. Binary/Choice/Score request validation, including Score requiring at least two levels.
2. Provider-neutral answer normalization.
3. Explicit provider+model decision resolution.
4. Decision provider failures become public GE failures, never SDK exceptions.
5. Observation model fields remain truthful when concrete upstream version is unknown.
6. Existing public text/image/structured tests remain green.
7. Public-surface export test covers the new types.

Acceptance token:

```text
GEJ_01_GENERIC_DECISION_CAPABILITY_ACCEPTED
```

## Stop conditions

Stop and revise architecture if implementing decisions requires:
- embedding product semantics into GE;
- parsing free-form generated prose as the decision contract;
- weakening existing provider/failure boundaries;
- pretending a gateway alias is a concrete upstream model version.
