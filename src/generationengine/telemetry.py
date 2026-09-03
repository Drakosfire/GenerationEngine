"""Bounded telemetry payloads. Do not put full prompts or generated content here."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from generationengine.models.requests import ImageGenerationRequest, TextGenerationRequest


def schema_hash(schema: dict[str, Any]) -> str:
    encoded = json.dumps(schema, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def bounded_text_metrics_input(request: TextGenerationRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "system_prompt_length": len(request.system_prompt or ""),
        "user_prompt_length": len(request.user_prompt),
        "model": request.model.value,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "schema_name": request.response_schema_name if request.response_schema else None,
    }
    if request.response_schema is not None:
        payload["schema_hash"] = schema_hash(request.response_schema)
    return payload


def bounded_image_metrics_input(request: ImageGenerationRequest) -> dict[str, Any]:
    return {
        "image_prompt_length": len(request.prompt),
        "model": request.model.value,
        "num_images": request.num_images,
        "size": request.size.value,
        "has_image_url": request.image_url is not None,
        "has_strength": request.strength is not None,
        "has_mask": request.mask_base64 is not None,
        "has_base_image": request.base_image_base64 is not None,
        "has_negative_prompt": request.negative_prompt is not None,
        "negative_prompt_length": len(request.negative_prompt or ""),
    }
