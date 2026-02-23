from __future__ import annotations

import argparse
import asyncio
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
        query_parameters=query_parameters,
    )
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

    task_prompt = DynamicResponseGenerator._build_prompt(
        operation=operation,
        validated_request_body=request_body,
        context_payload=context_payload,
        independent_requests=settings.generation_independent_requests,
    )
    full_prompt = generator._build_ollama_prompt(task_prompt)

    print("=== Request Debug ===")
    print(f"Spec: {spec_path}")
    print(f"Operation: {operation.method.upper()} {operation.path}")
    print(f"Resolved path params: {json_dumps(path_parameters)}")
    print(f"Query params: {json_dumps(query_parameters)}")
    print(f"Independent mode: {settings.generation_independent_requests}")

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

    raw_output = data.get("response")
    if not isinstance(raw_output, str):
        raise ValueError(f"Unexpected Ollama response payload: {data}")

    print("\n=== Raw Ollama Output ===")
    _safe_print(raw_output)

    parsed = extract_json_value(raw_output)
    normalized = DynamicResponseGenerator._normalize_candidate_shape(operation.response_schema, parsed)
    shape_ok = DynamicResponseGenerator._matches_response_shape(operation.response_schema, normalized)
    synthesized = generator.schema_synthesizer.build(
        schema=operation.response_schema,
        request_body=request_body,
        path_parameters=path_parameters,
        collection=context_payload.get("collection"),
        source_candidate=normalized,
    )
    final_candidate = DynamicResponseGenerator._enforce_identifier_consistency(
        synthesized,
        context_payload,
    )

    print("\n=== Parsed Candidate (normalized) ===")
    _safe_print(json_dumps(normalized))
    print(f"\nShape match: {shape_ok}")

    print("\n=== Final Candidate (after synthesizer + identifier consistency) ===")
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
