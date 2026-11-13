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
    _project_provider_schema(schema, schema)

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




def _project_provider_schema(schema: Any, root: dict[str, Any]) -> None:
    """Project schema annotations and Pydantic tagged unions for OpenAI.

    Traverse schema positions only. Local validation retains the original
    schema, including exclusivity for tagged unions.
    """
    if not isinstance(schema, dict):
        return
    schema.pop("uniqueItems", None)
    schema.pop("default", None)
    if "discriminator" in schema and "oneOf" in schema and "anyOf" not in schema and _has_distinct_tags(schema, root):
        schema.pop("discriminator")
        schema["anyOf"] = schema.pop("oneOf")
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        children = schema.get(key)
        if isinstance(children, dict):
            for child in children.values():
                _project_provider_schema(child, root)
    for key in (
        "items", "additionalItems", "additionalProperties", "unevaluatedItems",
        "unevaluatedProperties", "contains", "propertyNames", "not", "if", "then", "else",
    ):
        child = schema.get(key)
        if isinstance(child, list):
            for item in child:
                _project_provider_schema(item, root)
        else:
            _project_provider_schema(child, root)
    for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                _project_provider_schema(child, root)


def _has_distinct_tags(schema: dict[str, Any], root: dict[str, Any]) -> bool:
    discriminator = schema["discriminator"]
    name = discriminator.get("propertyName") if isinstance(discriminator, dict) else None
    if not isinstance(name, str) or not isinstance(schema["oneOf"], list) or not schema["oneOf"]:
        return False

    tags = set()
    for branch in schema["oneOf"]:
        if not isinstance(branch, dict):
            return False
        ref = branch.get("$ref")
        if ref is not None:
            if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
                return False
            branch = root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
        if not isinstance(branch, dict) or branch.get("type") != "object":
            return False
        properties = branch.get("properties")
        field = properties.get(name) if isinstance(properties, dict) else None
        tag = _clean_schema_node(copy.deepcopy(field)).get("const") if isinstance(field, dict) else None
        if not isinstance(tag, str) or tag in tags:
            return False
        # The wire projection makes every declared object property required.
        tags.add(tag)
    return True
