"""A small JSON Schema validator for extraction schemas (type, properties, required, items,
enum, nullable types), and the repairs applied to model output before validation."""

from __future__ import annotations

import json
import re

_TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool,
          "null": type(None)}


def errors(value, schema: dict, path: str = "$") -> list[str]:
    out: list[str] = []
    types = schema.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        ok = any(isinstance(value, _TYPES[t]) and not (t in {"integer", "number"} and isinstance(value, bool))
                 for t in allowed)
        if not ok:
            return [f"{path}: expected {' or '.join(allowed)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                out.append(f"{path}: missing {key}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                out += errors(value[key], sub, f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            out += [f"{path}: unexpected {k}" for k in value if k not in props]
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            out += errors(item, schema["items"], f"{path}[{i}]")
    return out


def repair(text: str) -> str:
    """Undo the usual ways a model breaks JSON: code fences, prose around the object, trailing
    commas, Python literals."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", text))
    return text


def parse(text: str, schema: dict) -> tuple[dict | None, list[str]]:
    """Parse as-is; if that fails, repair once and try again. Returns (value, errors)."""
    for attempt in (text, repair(text)):
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError as exc:
            last = [f"not JSON: {exc.msg}"]
            continue
        errs = errors(value, schema)
        if not errs:
            return value, []
        last = errs
    return None, last
