# GenerationEngine

Provider-agnostic inference execution for DungeonMind products.

GenerationEngine owns inference execution and inference-call truth. Products own prompts, schemas, workflows, and artifact persistence.

| Document | Role |
| --- | --- |
| [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md) | Implemented public surface |
| [docs/CORE-CONTRACT.md](docs/CORE-CONTRACT.md) | Capabilities, profiles, observations, failures |
| [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) | Deleted vs current API |
| [docs/E2-SUCCESSOR-SLICES.md](docs/E2-SUCCESSOR-SLICES.md) | E2 sequence |

## Installation

```bash
pip install "generationengine[openai,fal] @ git+https://github.com/Drakosfire/GenerationEngine.git"
```

Text requires `OPENAI_API_KEY`. Images require `FAL_KEY`. Cloudflare credentials are not used here.

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
FAL_KEY=your_fal_key
```
