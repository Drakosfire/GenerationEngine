# GenerationEngine

Provider-agnostic inference execution for DungeonMind products.

GenerationEngine owns inference execution and inference-call truth. Products own prompts, schemas, workflows, and artifact persistence.

| Document | Role |
| --- | --- |
| [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md) | Implemented public surface |
| [docs/CORE-CONTRACT.md](docs/CORE-CONTRACT.md) | Capabilities, profiles, observations, failures |
| [docs/STRUCTURED-CONFORMANCE.md](docs/STRUCTURED-CONFORMANCE.md) | Adopted structured-generation refinement |
| [docs/README.md](docs/README.md) | Documentation authority/index |

## Installation

```bash
pip install "generationengine[openai,openrouter,fal,typesafe] @ git+https://github.com/Drakosfire/GenerationEngine.git"
```

Text through OpenAI requires `OPENAI_API_KEY`. Text through OpenRouter requires `OPENROUTER_API_KEY`. Images require `FAL_KEY`. TypeSafe decisions require `TYPESAFE_JEV_API_KEY`; the adapter passes it directly to the official async SDK over the Vercel AI Gateway route. Cloudflare credentials are not used here.

## Quick start

```python
from generationengine import GenerationClient, InferenceProfile, TextRequest

client = GenerationClient.from_env()
try:
    result = await client.generate_text(
        TextRequest(
            user_prompt="What is a statblock?",
            profile=InferenceProfile.TEXT_FAST,
        )
    )
    print(result.text)
    print(result.observation.resolved_model)
finally:
    await client.aclose()
```

Explicit provider+model targets do not require a catalog row. The named provider must be registered:

```python
client = GenerationClient.from_env()
try:
    result = await client.generate_text(
        TextRequest(
            user_prompt="Summarize this scene",
            provider="openrouter",
            model="deepseek/deepseek-v4.1-flash",
        )
    )
    print(result.observation.provider)
    print(result.observation.resolved_model)
finally:
    await client.aclose()
```

Structured generation:

```python
client = GenerationClient.from_env()
try:
    result = await client.generate_structured(
        TextRequest(
            user_prompt="Generate a creature named Bob",
            profile=InferenceProfile.STRUCTURED_HIGH_RELIABILITY,
            json_schema=schema,
            schema_name="Creature",
        )
    )
    print(result.parsed)
finally:
    await client.aclose()
```

`generate_structured()` performs GenerationEngine-owned local schema validation and at most one structural repair. OpenAI native JSON Schema is an optimization, not the contract; see [docs/STRUCTURED-CONFORMANCE.md](docs/STRUCTURED-CONFORMANCE.md). OpenRouter uses JSON instructions rather than native `json_schema`.

Typed decisions use the separate `decide()` method. The provider adapter owns SDK mapping and transport; the caller owns state, question wording, and label meaning:

```python
from generationengine import BinaryDecisionQuestion, DecisionRequest

client = GenerationClient.from_env()
try:
    result = await client.decide(DecisionRequest(
        state={"text": "A sample passage"},
        questions=(BinaryDecisionQuestion(name="relevant", question="Is it relevant?"),),
        provider="typesafe",
        model="typesafe-ai/jev",
    ))
    print(result.answers["relevant"].probability_true)
    print(result.observation.provider_transport)
finally:
    await client.aclose()
```

Run the opt-in live boundary witness with `uv run --extra typesafe --env-file /path/to/.env.development python scripts/smoke_typesafe_decision.py`. It prints typed answers and model/transport/usage metadata without printing credentials.

Ordinary text also supports provider-neutral `max_output_tokens` and schema-less `json_object=True`. JSON-object mode returns raw text with `parsed=None`; products retain parsing/domain/fallback ownership. Labs that need provider-specific controls GE cannot express may still use a bounded direct path.

Image generation returns bytes. Products publish artifacts:

```python
from generationengine import ImageRequest

client = GenerationClient.from_env()
try:
    images = await client.generate_image(
        ImageRequest(
            prompt="A mystical dragon in a forest",
            profile=InferenceProfile.IMAGE_HIGH_QUALITY,
            model="gpt-image-1.5",
        )
    )
    png_bytes = images.images[0].content
finally:
    await client.aclose()
```

## Environment variables

```bash
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
FAL_KEY=your_fal_key
TYPESAFE_JEV_API_KEY=your_gateway_key
```
