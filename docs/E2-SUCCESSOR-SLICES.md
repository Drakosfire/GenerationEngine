# E2 successor slices

E2A is characterization and contract design only. Implementation follows in attributable slices. Do not merge E2B–E2E into one PR.

---

## E2B — Trustworthy core primitives and repository baseline

```text
.gitignore and stop tracking bytecode
GitHub Actions: lock check, pytest, ruff, build on a clean install
move pytest off the runtime dependency list
installable extras skeleton (openai / fal / dev)
fix the 4 pre-existing baseline test failures (RateLimitError mock; MagicMock usage JSON)
normalized failure types matching CORE-CONTRACT §6
InferenceObservation type matching CORE-CONTRACT §5
model/catalog/pricing authority skeleton (shape + empty/minimal records, not a product policy dump)
generic profile vocabulary as data, not product names
TextProvider / ImageProvider protocols in core (no SDK rewrite yet)
consumer-required imports and return shapes remain importable
unused baseline exports (IGenerator, GenerationResponse, RetryableError, …) may be retired
```

No consumer migration. No live provider wiring change.

---

## E2C — Provider-neutral text execution

```text
move AsyncOpenAI behind TextProvider
preserve TextGenerationService / TextGenerationRequest / TextModel as consumer facades
transport-neutral stream events internally
optional SSE adapter only if useful; generate_stream has no Server caller
normalize usage, request IDs, failures, latency, retry_count
remove MODEL_PRICING from the text service; read the catalog
keep current GPT-5.1 effective selection for compatibility callers unless a facade test proves otherwise
```

No DungeonMindServer migration unless a tiny in-repo compatibility proof requires it. E3 owns real consumers.

---

## E2D — Artifact-decoupled image execution

```text
image generate/edit returns GeneratedImage without Cloudflare
make advertised provider wiring truthful (Fal extra + OpenAI image provider registration policy)
fal-client as a real optional extra
ImageService URL-returning path remains as compatibility adapter
normalized image observations and failures
text-only construction does not require Fal or Cloudflare
```

Do not opportunistically change which Fal model IDs current Server callers use.

---

## E2E — GenerationEngine internal readiness

```text
fresh-install capability proof (text extra, image extra, both, neither)
fake-provider contract tests
text + structured + stream + image/edit tests
pricing/cost tests including unknown vs zero
failure/retry tests
no product-domain vocabulary in core profiles/contracts
README truthful: current compatibility API vs implemented target
```

After E2E, E3 migrates real DungeonMindServer paths. Settling Gate G follows E3 proofs.

---

## Attribution from E2A evidence

The default sequence above is confirmed, not replaced:

- Test/CI/hygiene rot is a baseline-trust problem → **E2B**, not the provider rewrite.
- Direct `AsyncOpenAI` and SSE framing are text-execution problems → **E2C**.
- Fal wiring vs README, missing `fal-client`, Cloudflare coupling → **E2D**.
- README/install truth and fake-provider proofs need the primitives first → **E2E**.

Do not start E3 until E2E says the package is worthy of real consumers.
