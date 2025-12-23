# GenerationEngine

Unified generation infrastructure for DungeonMind generators.

## Overview

GenerationEngine provides a consistent interface for AI-powered generation services across DungeonMind, including:

- **Image Generation**: Unified image generation via Fal.ai and OpenAI DALL-E
- **Text Generation**: Structured and streaming text generation via OpenAI Responses API
- **Metrics Tracking**: Built-in telemetry for generation performance and costs
- **Error Handling**: Robust retry logic with exponential backoff

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

response = await service.generate(request, service_name="myapp")
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
    model=TextModel.GPT_4O,
    temperature=0.7
)

response = await service.generate(request, service_name="myapp")
if response.success:
    print(f"Generated text: {response.content}")
```

### Streaming Text Generation

```python
async for chunk in service.generate_stream(request, service_name="myapp"):
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
    model=TextModel.GPT_4O,
    response_schema=schema,
    response_schema_name="Creature"
)

response = await service.generate(request, service_name="myapp")
if response.success and response.parsed_content:
    creature = Creature(**response.parsed_content)
    print(f"Generated: {creature.name} (level {creature.level})")
```

## Requirements

- Python >= 3.11
- OpenAI API key (for text generation)
- Fal.ai API key (optional, for image generation)
- Cloudflare API credentials (optional, for image uploads)

## Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
FAL_KEY=your_fal_key  # Optional
CLOUDFLARE_ACCOUNT_ID=your_account_id  # Optional
CLOUDFLARE_API_TOKEN=your_api_token  # Optional
```

## License

[Add license here]

## Links

- [GitHub Repository](https://github.com/Drakosfire/GenerationEngine)
- [Documentation](https://github.com/Drakosfire/GenerationEngine)

