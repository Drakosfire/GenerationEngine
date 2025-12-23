"""Schema transformation utilities for OpenAI Structured Outputs.

OpenAI Structured Outputs requires schemas to meet specific requirements:
1. All objects must have 'additionalProperties: false'
2. All properties must be in the 'required' array
3. Optional fields should be handled by allowing null in their type
4. $ref cannot have additional keywords (description, default, etc.)
5. $defs section must also be cleaned

This module transforms Pydantic-generated JSON schemas to meet these requirements.
"""

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def make_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Transform Pydantic schema for OpenAI structured outputs strict mode.

    Args:
        schema: Pydantic model's JSON schema (from model.model_json_schema())

    Returns:
        Schema transformed to meet OpenAI strict mode requirements
    """
    # Deep copy to avoid mutating original
    schema = copy.deepcopy(schema)

    # CRITICAL: Process $defs section FIRST
    # This is where referenced types live (like SpellSlots)
    if "$defs" in schema:
        logger.debug(f"Processing {len(schema['$defs'])} schema definitions")
        for def_name in list(schema["$defs"].keys()):
            schema["$defs"][def_name] = _clean_schema_node(schema["$defs"][def_name])

    # Then process the main schema
    schema = _clean_schema_node(schema)

    return schema


def _clean_schema_node(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Clean a single schema node (recursive helper for make_schema_strict).

    Handles:
    - Nullable $ref (anyOf with $ref and null)
    - $ref with extra keywords
    - Adding additionalProperties: false to objects
    - Making all properties required
    """
    if not isinstance(schema, dict):
        return schema

    # FIRST: Handle nullable $ref (anyOf with $ref and null) BEFORE other processing
    # This is a special case: {"anyOf": [{"$ref": "..."}, {"type": "null"}], "default": null, ...}
    # OpenAI strict mode doesn't allow $ref with sibling keywords
    if "anyOf" in schema:
        any_of_types = schema["anyOf"]
        non_null_types = [t for t in any_of_types if t.get("type") != "null"]
        has_null = any(t.get("type") == "null" for t in any_of_types)

        # Check if the non-null type is a $ref
        if has_null and len(non_null_types) == 1 and "$ref" in non_null_types[0]:
            # Nullable $ref - we need to keep anyOf structure
            # OpenAI doesn't allow {"$ref": "...", "default": null}
            # So we keep it as anyOf but clean up the structure
            return {
                "anyOf": [
                    {"$ref": non_null_types[0]["$ref"]},
                    {"type": "null"},
                ]
            }
        elif has_null and len(non_null_types) == 1:
            # Nullable non-$ref type - can flatten to type array
            actual_type = non_null_types[0]
            # Start fresh with just the actual type
            result = {}
            for key, value in actual_type.items():
                result[key] = value
            # Add null to type
            if "type" in result:
                result["type"] = [result["type"], "null"]
            return _clean_schema_node(result)

    # SECOND: Handle $ref with extra keywords - must be done before any other processing
    if "$ref" in schema and len(schema) > 1:
        # $ref must be alone - return a clean $ref
        logger.debug(f"Cleaning $ref with extra keys: {list(schema.keys())}")
        return {"$ref": schema["$ref"]}

    # THIRD: Recursively process nested schemas BEFORE modifying current schema
    # This ensures all nested $refs are cleaned up first
    for key, value in list(schema.items()):  # Use list() to avoid dict size change during iteration
        if isinstance(value, dict):
            schema[key] = _clean_schema_node(value)
        elif isinstance(value, list):
            schema[key] = [_clean_schema_node(item) if isinstance(item, dict) else item for item in value]

    # FOURTH: Add additionalProperties: false to objects
    if schema.get("type") == "object":
        schema["additionalProperties"] = False

        # Make sure all properties are required
        # OpenAI strict mode requires all properties to be in required array
        if "properties" in schema:
            all_props = list(schema["properties"].keys())
            schema["required"] = all_props

    return schema


