from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings
from app.context_engine import ContextEngine
from app.openapi_loader import OpenAPILoader, OperationMeta
from app.response_generator import DynamicResponseGenerator
from app.semantic_validator import SemanticValidator
from app.utils import extract_json_value, json_dumps


def _parse_json_value(raw: str, label: str) -> Any:
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}: {exc}") from exc


def _parse_json_object(raw: str, label: str) -> dict[str, Any]:
    parsed = _parse_json_value(raw, label)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def _safe_print(text: Any) -> None:
    value = str(text)
    encoding = sys.stdout.encoding or "utf-8"
    safe = value.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore")
    print(safe)


def _matches_response_shape_local(schema: dict[str, Any] | None, candidate: Any) -> bool:
    if not isinstance(schema, dict):
        return True
    expected_type = schema.get("type")
    if expected_type == "object":
        return isinstance(candidate, dict)
    if expected_type == "array":
        return isinstance(candidate, list)
    return True


def _normalize_candidate_shape_local(schema: dict[str, Any] | None, candidate: Any) -> Any:
    if not isinstance(schema, dict):
        return candidate
    expected_type = schema.get("type")
    if expected_type == "array":
        if isinstance(candidate, list):
            return candidate
        if isinstance(candidate, dict):
            list_values = [value for value in candidate.values() if isinstance(value, list)]
            if len(list_values) == 1:
                return list_values[0]
            if len(list_values) > 1:
                return max(list_values, key=len)
            return [candidate]
    if expected_type == "object":
        if isinstance(candidate, list) and len(candidate) == 1 and isinstance(candidate[0], dict):
            return candidate[0]
    return candidate


def _compile_path_template(template: str) -> tuple[re.Pattern[str], list[str]]:
    names: list[str] = []
    parts: list[str] = []
    for token in re.split(r"(\{[^{}]+\})", template):
        if not token:
            continue
        if token.startswith("{") and token.endswith("}"):
            name = token[1:-1].strip()
            names.append(name)
            parts.append(f"(?P<{name}>[^/]+)")
        else:
            parts.append(re.escape(token))
    pattern = "^" + "".join(parts) + "$"
    return re.compile(pattern), names


def _resolve_operation(
    operations: list[OperationMeta],
    method: str,
    requested_path: str,
) -> tuple[OperationMeta, dict[str, Any]]:
    method_lc = method.strip().lower()
    exact = [op for op in operations if op.method == method_lc and op.path == requested_path]
    if exact:
        return exact[0], {}

    for op in operations:
        if op.method != method_lc:
            continue
        regex, names = _compile_path_template(op.path)
        match = regex.match(requested_path)
        if match is None:
            continue
        extracted: dict[str, Any] = {}
        for name in names:
            raw_value = match.group(name)
            if raw_value.isdigit():
                extracted[name] = int(raw_value)
            else:
                extracted[name] = raw_value
        return op, extracted

    available = ", ".join(sorted({f"{op.method.upper()} {op.path}" for op in operations}))
    raise ValueError(
        f"No matching operation for method={method.upper()} path={requested_path}. "
        f"Available: {available}"
    )


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise ValueError(f"Spec file not found: {spec_path}")

    raw_spec = spec_path.read_text(encoding="utf-8")
    loader = OpenAPILoader()
    spec_dict, _, operations = loader.load(raw_spec)
    operation, extracted_params = _resolve_operation(
        operations=operations,
        method=args.method,
        requested_path=args.path,
    )

    request_body = _parse_json_value(args.body, "body")
    path_parameters = _parse_json_object(args.path_params, "path-params")
    query_parameters = _parse_json_object(args.query_params, "query-params")
    for key, value in extracted_params.items():
        path_parameters.setdefault(key, value)

    context_engine = ContextEngine(history_limit=settings.context_history_limit)
    context_payload = context_engine.get_context_for_prompt(
        path_template=operation.path,
        method=operation.method,
        path_parameters=path_parameters,
    )
    context_payload["query_parameters"] = query_parameters
    info = spec_dict.get("info", {}) if isinstance(spec_dict, dict) else {}
    if not isinstance(info, dict):
        info = {}
    context_payload["api_info"] = {
        "title": info.get("title"),
        "description": info.get("description"),
        "version": info.get("version"),
    }
    context_payload["operation_summary"] = operation.summary
    context_payload["operation_description"] = operation.description
    context_payload["request_nonce"] = time.time_ns()

    generator = DynamicResponseGenerator(
        settings=settings,
        semantic_validator=SemanticValidator(
            model_name=settings.semantic_model_name,
            threshold=settings.semantic_similarity_threshold,
        ),
        seq2seq_generator=None,
    )

    independent_mode = bool(getattr(settings, "generation_independent_requests", False))
    build_prompt_signature = inspect.signature(DynamicResponseGenerator._build_prompt)
    prompt_kwargs: dict[str, Any] = {
        "operation": operation,
        "validated_request_body": request_body,
        "context_payload": context_payload,
    }
    if "independent_requests" in build_prompt_signature.parameters:
        prompt_kwargs["independent_requests"] = independent_mode
    if "attempt_number" in build_prompt_signature.parameters:
        prompt_kwargs["attempt_number"] = 1
    if "max_attempts" in build_prompt_signature.parameters:
        prompt_kwargs["max_attempts"] = settings.max_retry_attempts
    if "previous_failures" in build_prompt_signature.parameters:
        prompt_kwargs["previous_failures"] = []
    if "blocked_literals" in build_prompt_signature.parameters:
        prompt_kwargs["blocked_literals"] = []
    task_prompt = DynamicResponseGenerator._build_prompt(**prompt_kwargs)
    full_prompt = generator._build_ollama_prompt(task_prompt)

    print("=== Request Debug ===")
    print(f"Spec: {spec_path}")
    print(f"Operation: {operation.method.upper()} {operation.path}")
    print(f"Resolved path params: {json_dumps(path_parameters)}")
    print(f"Query params: {json_dumps(query_parameters)}")
    print(f"Independent mode: {independent_mode}")

    print("\n=== Prompt ===")
    if args.show_prompt:
        _safe_print(full_prompt)
    else:
        preview = full_prompt[:2200]
        suffix = "" if len(full_prompt) <= 2200 else "\n... (truncated, use --show-prompt for full text)"
        _safe_print(preview + suffix)

    selected_temperature = (
        settings.generation_temperature
        if args.temperature is None
        else float(args.temperature)
    )
    payload = {
        "model": args.model or settings.ollama_model,
        "prompt": full_prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": selected_temperature,
            "num_predict": settings.ollama_num_predict,
            "top_k": settings.ollama_top_k,
            "top_p": settings.ollama_top_p,
            "repeat_penalty": settings.ollama_repeat_penalty,
            "seed": int(time.time_ns() % 2147483647),
        },
    }

    target_url = args.url or settings.ollama_url
    timeout = httpx.Timeout(settings.generation_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            target_url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    print(f"Ollama model (requested): {payload['model']}")
    print(f"Ollama model (response): {data.get('model')}")

    raw_output = data.get("response")
    if not isinstance(raw_output, str):
        raise ValueError(f"Unexpected Ollama response payload: {data}")

    print("\n=== Raw Ollama Output ===")
    _safe_print(raw_output)

    parsed = extract_json_value(raw_output)
    normalize_fn = getattr(DynamicResponseGenerator, "_normalize_candidate_shape", None)
    if callable(normalize_fn):
        normalized = normalize_fn(operation.response_schema, parsed)
    else:
        normalized = _normalize_candidate_shape_local(operation.response_schema, parsed)

    shape_fn = getattr(DynamicResponseGenerator, "_matches_response_shape", None)
    if callable(shape_fn):
        shape_ok = shape_fn(operation.response_schema, normalized)
    else:
        shape_ok = _matches_response_shape_local(operation.response_schema, normalized)

    final_candidate = normalized
    used_schema_fallback = False
    placeholder_check_fn = getattr(generator, "_has_generic_placeholder_content", None)
    has_placeholder_content = (
        bool(placeholder_check_fn(final_candidate))
        if callable(placeholder_check_fn)
        else False
    )

    print("\n=== Parsed Candidate (normalized) ===")
    _safe_print(json_dumps(normalized))
    print(f"\nShape match: {shape_ok}")
    print(f"Schema fallback used: {used_schema_fallback}")
    print(f"Has generic placeholder content: {has_placeholder_content}")

    print("\n=== Final Candidate (LLM output after normalization only) ===")
    _safe_print(json_dumps(final_candidate))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send the same server prompt to Ollama for one OpenAPI operation and print raw output.",
    )
    parser.add_argument("--spec", required=True, help="Path to OpenAPI YAML/JSON file.")
    parser.add_argument("--path", required=True, help="Operation path template or concrete path.")
    parser.add_argument("--method", default="get", help="HTTP method (default: get).")
    parser.add_argument("--path-params", default="{}", help="JSON object for path parameters.")
    parser.add_argument("--query-params", default="{}", help="JSON object for query parameters.")
    parser.add_argument("--body", default="null", help="JSON value for validated request body.")
    parser.add_argument("--temperature", type=float, default=None, help="Override generation temperature.")
    parser.add_argument("--model", default=None, help="Override Ollama model.")
    parser.add_argument("--url", default=None, help="Override Ollama URL.")
    parser.add_argument("--show-prompt", action="store_true", help="Print full prompt instead of preview.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
