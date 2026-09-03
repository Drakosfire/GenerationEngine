# GenerationEngine

Provider-agnostic inference execution for DungeonMind products.

## Status (E2A)

This package is a **real in-process capability** with DungeonMindServer consumers, not a greenfield service.

E2A defines the target product-neutral contract. It does **not** implement that contract. The examples below match the **implemented** compatibility API as of baseline `f6f0fa95` (not the E2 target). Remaining claimed-vs-wired gaps (unwired OpenAI image provider, Fal packaging) are in [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md).

| Document | Role |
| --- | --- |
| [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md) | Claimed vs implemented; consumer inventory |
| [docs/CORE-CONTRACT.md](docs/CORE-CONTRACT.md) | Target capabilities, profiles, observations, failures |
| [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) | Consumer seams vs historical `__all__` |
| [docs/E2-SUCCESSOR-SLICES.md](docs/E2-SUCCESSOR-SLICES.md) | E2B–E2E sequence |

GenerationEngine owns inference execution and inference-call truth. Products own prompts, schemas, workflows, and artifact persistence.

## Overview

Current compatibility capabilities:

- **Image Generation**: Fal.ai models via `ImageService` (OpenAI image provider exists but is not wired; Cloudflare upload is currently mandatory)
- **Text Generation**: Structured and streaming text via OpenAI Responses API (`AsyncOpenAI` is constructed directly)
- **Metrics Tracking**: In-memory `GenerationMetrics` stub (not yet `InferenceObservation`)
- **Error Handling**: Retry with exponential backoff around current error codes

## Installation

### From PyPI (when published)

```bash
pip install generationengine
```

### From Git (development)

```bash
pip install git+https://github.com/Drakosfire/GenerationEngine.git
```

### Editable install (development)

```bash
git clone https://github.com/Drakosfire/GenerationEngine.git
cd GenerationEngine
uv pip install -e .
```

## Quick Start

### Image Generation

```python
from generationengine import ImageService, ImageGenerationRequest, ImageModel, ImageSize

service = ImageService()

request = ImageGenerationRequest(
    prompt="A mystical dragon in a forest",
    model=ImageModel.FLUX_PRO,
    size=ImageSize.SQUARE,
    num_images=1
)

response = await service.generate(request)
if response.success:
    print(f"Generated image: {response.images[0].url}")
```

### Text Generation

```python
from generationengine import TextGenerationService, TextGenerationRequest, TextModel

service = TextGenerationService()

request = TextGenerationRequest(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is a statblock?",
    model=TextModel.GPT_5_1,
    temperature=0.7
)

response = await service.generate(request, service_name="myapp")
if response.success:
    print(f"Generated text: {response.content}")
```

### Streaming Text Generation

```python
async for chunk in service.generate_stream(request):
    print(chunk, end="", flush=True)
```

### Structured Outputs

```python
from pydantic import BaseModel

class Creature(BaseModel):
    name: str
    level: int

schema = Creature.model_json_schema()

request = TextGenerationRequest(
    user_prompt="Generate a creature named Bob at level 5",
    model=TextModel.GPT_5_1,
    response_schema=schema,
    response_schema_name="Creature"
)

response = await service.generate(request, service_name="myapp")
if response.success and response.parsed_content:
    creature = Creature(**response.parsed_content)
    print(f"Generated: {creature.name} (level {creature.level})")
```

## Requirements

Current constructor/runtime requirements (target optionality is E2D, not current behavior):

- Python >= 3.11
- OpenAI API key to construct `TextGenerationService`
- Cloudflare Images credentials to construct `ImageService` / `UploadService`
- Fal.ai API key for `ImageService` to register any image providers (`fal-client` is currently undeclared; consumers often supply it)

## Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
FAL_KEY=your_fal_key
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_IMAGES_API_TOKEN=your_api_token
```

## License

[Add license here]

## Links

- [GitHub Repository](https://github.com/Drakosfire/GenerationEngine)
- [Documentation](https://github.com/Drakosfire/GenerationEngine)

