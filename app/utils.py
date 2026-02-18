import json
from copy import deepcopy
from typing import Any


HTTP_METHODS = (
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "options",
    "head",
)


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clamp_multi_response_count(query_value: Any, body_value: Any, max_n: int) -> int:
    selected = coerce_int(query_value)
    if selected is None:
        selected = coerce_int(body_value)
    if selected is None:
        return 1
    return max(1, min(selected, max_n))


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_value(text: str) -> Any:
    cleaned = strip_markdown_fences(text)
    decoder = json.JSONDecoder()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    valid_starts = set('{["-0123456789tfn')
    for idx, char in enumerate(cleaned):
        if char not in valid_starts:
            continue
        snippet = cleaned[idx:]
        try:
            value, _ = decoder.raw_decode(snippet)
            return value
        except json.JSONDecodeError:
            continue

    raise ValueError("LLM output did not contain a valid JSON value.")


def resolve_json_pointer(document: dict[str, Any], pointer: str) -> Any | None:
    if not pointer.startswith("#/"):
        return None

    current: Any = document
    parts = pointer[2:].split("/")
    for raw_part in parts:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def resolve_schema_refs(schema: Any, root_document: dict[str, Any], max_depth: int = 24) -> Any:
    return _resolve_refs(schema, root_document, seen_refs=set(), depth=0, max_depth=max_depth)


def _resolve_refs(
    node: Any,
    root_document: dict[str, Any],
    seen_refs: set[str],
    depth: int,
    max_depth: int,
) -> Any:
    if depth > max_depth:
        return node

    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in seen_refs:
                return node
            target = resolve_json_pointer(root_document, ref)
            if target is None:
                return node
            resolved = deepcopy(target)
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            if isinstance(resolved, dict) and siblings:
                resolved.update(siblings)
            return _resolve_refs(
                resolved,
                root_document,
                seen_refs=seen_refs | {ref},
                depth=depth + 1,
                max_depth=max_depth,
            )
        return {
            key: _resolve_refs(
                value,
                root_document,
                seen_refs=seen_refs,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for key, value in node.items()
        }

    if isinstance(node, list):
        return [
            _resolve_refs(
                item,
                root_document,
                seen_refs=seen_refs,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for item in node
        ]

    return node

