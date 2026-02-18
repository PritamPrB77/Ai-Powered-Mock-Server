from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import yaml
from openapi_core import OpenAPI
from openapi_spec_validator import validate

from app.utils import HTTP_METHODS, resolve_schema_refs


@dataclass(slots=True)
class OperationMeta:
    path: str
    method: str
    operation_id: str
    summary: str
    description: str
    request_schema: dict[str, Any] | None
    response_schema: dict[str, Any] | None
    response_status_code: int
    response_media_type: str | None


class OpenAPILoader:
    def load(self, raw_payload: str | bytes | dict[str, Any]) -> tuple[dict[str, Any], OpenAPI, list[OperationMeta]]:
        spec_dict = self._parse_payload(raw_payload)
        self._validate_spec(spec_dict)
        openapi = OpenAPI.from_dict(spec_dict)
        operations = self._extract_operations(spec_dict)
        return spec_dict, openapi, operations

    def _parse_payload(self, raw_payload: str | bytes | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw_payload, dict):
            return raw_payload

        text = raw_payload.decode("utf-8") if isinstance(raw_payload, bytes) else raw_payload
        stripped = text.strip()
        if not stripped:
            raise ValueError("OpenAPI payload is empty.")

        json_error: Exception | None = None
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:  # noqa: BLE001
            json_error = exc

        try:
            parsed_yaml = yaml.safe_load(stripped)
        except Exception as exc:  # noqa: BLE001
            if json_error is not None:
                raise ValueError("Payload is neither valid JSON nor valid YAML.") from exc
            raise

        if not isinstance(parsed_yaml, dict):
            raise ValueError("Parsed payload is not an OpenAPI object.")
        return parsed_yaml

    @staticmethod
    def _validate_spec(spec_dict: dict[str, Any]) -> None:
        validate(spec_dict)

    def _extract_operations(self, spec_dict: dict[str, Any]) -> list[OperationMeta]:
        operations: list[OperationMeta] = []
        paths = spec_dict.get("paths", {})
        if not isinstance(paths, dict):
            raise ValueError("OpenAPI spec has invalid 'paths' section.")

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method in HTTP_METHODS:
                operation = path_item.get(method)
                if not isinstance(operation, dict):
                    continue

                request_schema = self._extract_request_schema(operation, spec_dict)
                (
                    response_schema,
                    response_status_code,
                    response_media_type,
                ) = self._extract_response_schema(operation, spec_dict)

                operation_id = operation.get("operationId") or self._build_operation_id(path, method)
                operations.append(
                    OperationMeta(
                        path=path,
                        method=method,
                        operation_id=operation_id,
                        summary=str(operation.get("summary", "")),
                        description=str(operation.get("description", "")),
                        request_schema=request_schema,
                        response_schema=response_schema,
                        response_status_code=response_status_code,
                        response_media_type=response_media_type,
                    )
                )

        return operations

    def _extract_request_schema(
        self,
        operation: dict[str, Any],
        spec_dict: dict[str, Any],
    ) -> dict[str, Any] | None:
        request_body = operation.get("requestBody")
        if not isinstance(request_body, dict):
            return None

        resolved_request_body = resolve_schema_refs(request_body, spec_dict)
        content = resolved_request_body.get("content", {})
        if not isinstance(content, dict) or not content:
            return None

        media_type = "application/json" if "application/json" in content else next(iter(content.keys()))
        media_obj = content.get(media_type, {})
        if not isinstance(media_obj, dict):
            return None

        schema = media_obj.get("schema")
        if schema is None:
            return None
        resolved_schema = resolve_schema_refs(schema, spec_dict)
        return resolved_schema if isinstance(resolved_schema, dict) else None

    def _extract_response_schema(
        self,
        operation: dict[str, Any],
        spec_dict: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, int, str | None]:
        responses = operation.get("responses", {})
        if not isinstance(responses, dict) or not responses:
            return None, 200, None

        ordered_codes = list(responses.keys())
        preferred_code = self._pick_response_code(ordered_codes)
        selected = responses.get(preferred_code)
        if not isinstance(selected, dict):
            return None, self._status_code_from_key(preferred_code), None

        resolved_response = resolve_schema_refs(selected, spec_dict)
        content = resolved_response.get("content", {})
        if not isinstance(content, dict) or not content:
            return None, self._status_code_from_key(preferred_code), None

        media_type = "application/json" if "application/json" in content else next(iter(content.keys()))
        media_obj = content.get(media_type, {})
        if not isinstance(media_obj, dict):
            return None, self._status_code_from_key(preferred_code), media_type

        schema = media_obj.get("schema")
        resolved_schema = (
            resolve_schema_refs(schema, spec_dict)
            if isinstance(schema, dict)
            else None
        )

        return (
            resolved_schema if isinstance(resolved_schema, dict) else None,
            self._status_code_from_key(preferred_code),
            media_type,
        )

    @staticmethod
    def _pick_response_code(codes: list[str]) -> str:
        for code in codes:
            if code.startswith("2"):
                return code
        if "default" in codes:
            return "default"
        return codes[0]

    @staticmethod
    def _status_code_from_key(code: str) -> int:
        if code.isdigit():
            return int(code)
        if code.startswith("2"):
            return 200
        return 200

    @staticmethod
    def _build_operation_id(path: str, method: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_")
        return f"{method}_{normalized or 'root'}"

