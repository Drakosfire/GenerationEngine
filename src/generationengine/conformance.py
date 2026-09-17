"""GenerationEngine-owned structural conformance for generate_structured().

Provider-native schema features are optimizations. Local parse and JSON Schema
validation are the public structured contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from generationengine.providers.base import TextGenerationResult

MAX_REPAIR_ERRORS = 8
MAX_ERROR_MESSAGE_CHARS = 160
MAX_SCHEMA_CHARS = 16_384

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass
class ConformanceError(Exception):
    """Structural mismatch. Must not include product/domain judgments."""

    reason: str
    errors: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.reason


def candidate_from_result(result: TextGenerationResult) -> Any:
    if isinstance(result.parsed, dict):
        return result.parsed
    return parse_json_object(result.text)


def parse_json_object(text: str | None) -> Any:
    if text is None:
        raise ConformanceError("output was empty", ())
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ConformanceError(f"output was not valid JSON: {exc.msg}", ()) from exc
    if not isinstance(value, dict):
        raise ConformanceError("structured result must be a JSON object", ())
    return value


def check_schema(schema: dict[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ConformanceError(f"caller schema is not a valid JSON Schema: {exc.message}", ()) from exc


def validate_against_schema(instance: Any, schema: dict[str, Any]) -> None:
    try:
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=lambda err: list(err.absolute_path))
    except SchemaError as exc:
        raise ConformanceError(f"caller schema is not a valid JSON Schema: {exc.message}", ()) from exc
    if not errors:
        return
    formatted = tuple(_format_error(error) for error in errors[:MAX_REPAIR_ERRORS])
    raise ConformanceError("output did not satisfy the required schema", formatted)


def repair_instruction(failure: ConformanceError) -> str:
    lines = [
        "The previous output did not satisfy the required schema.",
        "Validation errors:",
    ]
    if failure.errors:
        lines.extend(failure.errors)
    else:
        lines.append(f"- $: {failure.reason}")
    lines.append("Return corrected JSON that satisfies the supplied schema.")
    return "\n".join(lines)


def schema_instruction(schema: dict[str, Any]) -> str:
    dumped = json.dumps(schema, ensure_ascii=True)
    if len(dumped) > MAX_SCHEMA_CHARS:
        dumped = dumped[: MAX_SCHEMA_CHARS - 3] + "..."
    return (
        "Respond with a JSON object only that satisfies this JSON Schema. "
        "Do not include markdown fences.\n"
        f"{dumped}"
    )


def _format_error(error: ValidationError) -> str:
    path = "$"
    for part in error.absolute_path:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    message = error.message.replace("\n", " ")
    if len(message) > MAX_ERROR_MESSAGE_CHARS:
        message = message[: MAX_ERROR_MESSAGE_CHARS - 3] + "..."
    return f"- {path}: {message}"
