# GenerationEngine

Provider-agnostic inference execution for DungeonMind products.

GenerationEngine owns inference execution and inference-call truth. Products own prompts, schemas, workflows, and artifact persistence.

| Document | Role |
| --- | --- |
| [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md) | Implemented public surface |
| [docs/CORE-CONTRACT.md](docs/CORE-CONTRACT.md) | Capabilities, profiles, observations, failures |
| [docs/STRUCTURED-CONFORMANCE.md](docs/STRUCTURED-CONFORMANCE.md) | Adopted structured-generation refinement |
| [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) | Deleted vs current API |
| [docs/E2-SUCCESSOR-SLICES.md](docs/E2-SUCCESSOR-SLICES.md) | E2 sequence |

## Installation

```bash
pip install "generationengine[openai,openrouter,fal] @ git+https://github.com/Drakosfire/GenerationEngine.git"
```

Text through OpenAI requires `OPENAI_API_KEY`. Text through OpenRouter requires `OPENROUTER_API_KEY`. Images require `FAL_KEY`. Cloudflare credentials are not used here.

## Quick start

```python
from generationengine import GenerationClient, InferenceProfile, TextRequest

client = GenerationClient.from_env()
result = await client.generate_text(
    TextRequest(
        user_prompt="What is a statblock?",
        profile=InferenceProfile.TEXT_FAST,
    )
)
print(result.text)
print(result.observation.resolved_model)
```

Explicit provider+model targets do not require a catalog row. The named provider must be registered:

```python
result = await client.generate_text(
    TextRequest(
        user_prompt="Summarize this scene",
        provider="openrouter",
        model="deepseek/deepseek-v4.1-flash",
    )
)
print(result.observation.provider)
print(result.observation.resolved_model)
```

Structured generation:

```python
result = await client.generate_structured(
    TextRequest(
        user_prompt="Generate a creature named Bob",
        profile=InferenceProfile.STRUCTURED_HIGH_RELIABILITY,
        json_schema=schema,
        schema_name="Creature",
    )
)
print(result.parsed)
```

`generate_structured()` performs GenerationEngine-owned local schema validation. OpenAI native JSON Schema is an optimization, not the contract; see [docs/STRUCTURED-CONFORMANCE.md](docs/STRUCTURED-CONFORMANCE.md). OpenRouter uses JSON instructions rather than native `json_schema`. Labs that need provider-specific routing/reasoning knobs GE cannot express may still use a bounded direct path.

Image generation returns bytes. Products publish artifacts:

```python
from generationengine import ImageRequest

images = await client.generate_image(
    ImageRequest(
        prompt="A mystical dragon in a forest",
        profile=InferenceProfile.IMAGE_HIGH_QUALITY,
        model="gpt-image-1.5",
    )
)
png_bytes = images.images[0].content
```

## Environment variables

```bash
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
FAL_KEY=your_fal_key
```
