from __future__ import annotations

import random
import re
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.field_semantics import IdentifierSemanticAnalyzer
from app.utils import coerce_int, json_dumps


class SchemaSynthesizer:
    FIRST_NAMES = (
        "Ava",
        "Liam",
        "Noah",
        "Emma",
        "Mia",
        "Arjun",
        "Zara",
        "Olivia",
        "Ethan",
        "Sophia",
    )
    LAST_NAMES = (
        "Shaw",
        "Bennett",
        "Nguyen",
        "Patel",
        "Singh",
        "Carter",
        "Mehta",
        "Lopez",
        "Diaz",
        "Khan",
    )
    EMAIL_DOMAINS = ("example.com", "mail.test", "demo.local")

    def build(
        self,
        schema: dict[str, Any] | None,
        request_body: Any,
        path_parameters: dict[str, Any] | None,
        collection: str | None = None,
        source_candidate: Any | None = None,
    ) -> Any | None:
        if not isinstance(schema, dict):
            return source_candidate

        request_obj = request_body if isinstance(request_body, dict) else {}
        path_obj = path_parameters if isinstance(path_parameters, dict) else {}
        rng = self._make_rng(request_obj, path_obj, source_candidate)
        return self._build_node(
            schema=schema,
            field_name=None,
            source_value=source_candidate,
            request_body=request_obj,
            path_parameters=path_obj,
            collection=collection,
            rng=rng,
        )

    def _make_rng(
        self,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
        source_candidate: Any,
    ) -> random.Random:
        seed_input = {
            "request_body": request_body,
            "path_parameters": path_parameters,
            "source_candidate": source_candidate,
            "nonce": time.time_ns(),
        }
        seed = hash(json_dumps(seed_input))
        return random.Random(seed)

    def _build_node(
        self,
        schema: dict[str, Any],
        field_name: str | None,
        source_value: Any,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
        collection: str | None,
        rng: random.Random,
    ) -> Any:
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            if source_value in enum_values:
                return source_value
            if field_name and request_body.get(field_name) in enum_values:
                return request_body[field_name]
            return rng.choice(enum_values)

        node_type = schema.get("type")
        if not node_type:
            if "properties" in schema:
                node_type = "object"
            elif "items" in schema:
                node_type = "array"

        if node_type == "object":
            return self._build_object(
                schema=schema,
                source_value=source_value,
                request_body=request_body,
                path_parameters=path_parameters,
                collection=collection,
                rng=rng,
            )

        if node_type == "array":
            return self._build_array(
                schema=schema,
                field_name=field_name,
                source_value=source_value,
                request_body=request_body,
                path_parameters=path_parameters,
                collection=collection,
                rng=rng,
            )

        if node_type == "string":
            return self._build_string(
                schema=schema,
                field_name=field_name,
                source_value=source_value,
                request_body=request_body,
                path_parameters=path_parameters,
                collection=collection,
                rng=rng,
            )

        if node_type == "integer":
            return self._build_integer(schema=schema, source_value=source_value, rng=rng)

        if node_type == "number":
            return self._build_number(schema=schema, source_value=source_value, rng=rng)

        if node_type == "boolean":
            if isinstance(source_value, bool):
                return source_value
            return bool(rng.randint(0, 1))

        return source_value if source_value is not None else {}

    def _build_object(
        self,
        schema: dict[str, Any],
        source_value: Any,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
        collection: str | None,
        rng: random.Random,
    ) -> dict[str, Any]:
        props = schema.get("properties")
        if not isinstance(props, dict):
            return {}

        source_obj = source_value if isinstance(source_value, dict) else {}
        required = schema.get("required")
        required_list = required if isinstance(required, list) else []

        output: dict[str, Any] = {}
        keys = list(props.keys())
        for key in keys:
            prop_schema = props.get(key)
            if not isinstance(prop_schema, dict):
                continue

            source_selected = self._lookup_mapping_value(source_obj, key)
            request_selected = self._lookup_mapping_value(request_body, key)
            path_selected = self._lookup_mapping_value(path_parameters, key)

            if self._is_identifier_like_field(
                field_name=key,
                schema=prop_schema,
                request_body=request_body,
                path_parameters=path_parameters,
            ):
                selected_source = path_selected
                if selected_source is None:
                    selected_source = request_selected
                if selected_source is None:
                    selected_source = source_selected
            else:
                selected_source = source_selected
                if selected_source is None:
                    selected_source = request_selected
                if selected_source is None:
                    selected_source = path_selected

            output[key] = self._build_node(
                schema=prop_schema,
                field_name=key,
                source_value=selected_source,
                request_body=request_body,
                path_parameters=path_parameters,
                collection=collection,
                rng=rng,
            )

        for key in required_list:
            if key in output:
                continue
            prop_schema = props.get(key)
            if isinstance(prop_schema, dict):
                output[key] = self._build_node(
                    schema=prop_schema,
                    field_name=key,
                    source_value=None,
                    request_body=request_body,
                    path_parameters=path_parameters,
                    collection=collection,
                    rng=rng,
                )
        return output

    def _build_array(
        self,
        schema: dict[str, Any],
        field_name: str | None,
        source_value: Any,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
        collection: str | None,
        rng: random.Random,
    ) -> list[Any]:
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return []

        source_list = source_value if isinstance(source_value, list) else []
        if not source_list and field_name and isinstance(request_body.get(field_name), list):
            source_list = request_body[field_name]

        min_items = schema.get("minItems")
        target_count = min_items if isinstance(min_items, int) and min_items > 0 else 1
        if source_list:
            target_count = max(target_count, min(len(source_list), 3))

        output: list[Any] = []
        for idx in range(target_count):
            item_source = source_list[idx] if idx < len(source_list) else None
            output.append(
                self._build_node(
                    schema=items_schema,
                    field_name=field_name,
                    source_value=item_source,
                    request_body=request_body,
                    path_parameters=path_parameters,
                    collection=collection,
                    rng=rng,
                )
            )
        return output

    def _build_string(
        self,
        schema: dict[str, Any],
        field_name: str | None,
        source_value: Any,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
        collection: str | None,
        rng: random.Random,
    ) -> str:
        field = (field_name or "").lower()
        fmt = str(schema.get("format", "")).lower()

        if isinstance(source_value, str) and source_value.strip():
            if path_parameters and any(
                token in field
                for token in ("translation", "commentary", "sanskrit", "text", "content")
            ):
                return self._ensure_path_context_in_text(source_value, path_parameters)
            return source_value

        exact_id = self._infer_exact_identifier(field_name, request_body, path_parameters)
        generic_id = self._infer_any_identifier(request_body, path_parameters)

        if fmt == "date-time":
            return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if fmt == "date":
            return datetime.now(timezone.utc).date().isoformat()
        if fmt == "email" or "email" in field:
            local_name = self._safe_slug(
                request_body.get("name")
                or request_body.get("customerName")
                or f"user{generic_id or rng.randint(100, 999)}"
            )
            return f"{local_name}@{rng.choice(self.EMAIL_DOMAINS)}"
        if fmt == "uuid":
            return str(uuid4())

        if field_name:
            from_request = self._lookup_mapping_value(request_body, field_name)
            if from_request is not None:
                return str(from_request)
            from_path = self._lookup_mapping_value(path_parameters, field_name)
            if from_path is not None:
                return str(from_path)

        if self._is_identifier_like_field(
            field_name=field_name,
            schema=schema,
            request_body=request_body,
            path_parameters=path_parameters,
        ):
            return exact_id or self._generate_identifier(field_name, collection, rng)

        if "name" in field:
            if "first" in field:
                return rng.choice(self.FIRST_NAMES)
            if "last" in field:
                return rng.choice(self.LAST_NAMES)
            resource_title = self._resource_title(collection)
            if generic_id:
                return f"{resource_title} {generic_id}"
            return f"{rng.choice(self.FIRST_NAMES)} {rng.choice(self.LAST_NAMES)}"

        if any(token in field for token in ("note", "description", "summary", "title")):
            resource_title = self._resource_title(collection)
            return f"{resource_title} detail {rng.randint(1000, 9999)}"

        if path_parameters and any(
            token in field
            for token in ("translation", "commentary", "sanskrit", "text", "content")
        ):
            return f"{self._humanize_field(field_name)} for {self._path_context_hint(path_parameters)}"

        min_len = schema.get("minLength")
        base = f"value_{rng.randint(100, 999)}"
        if isinstance(min_len, int) and min_len > len(base):
            base = base + ("x" * (min_len - len(base)))
        return base

    @staticmethod
    def _build_integer(schema: dict[str, Any], source_value: Any, rng: random.Random) -> int:
        parsed = coerce_int(source_value)
        if parsed is not None:
            return parsed

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        low = int(minimum) if isinstance(minimum, (int, float)) else 1
        high = int(maximum) if isinstance(maximum, (int, float)) else (low + 20)
        if high < low:
            high = low
        return rng.randint(low, high)

    @staticmethod
    def _build_number(schema: dict[str, Any], source_value: Any, rng: random.Random) -> float:
        if isinstance(source_value, (int, float)) and not isinstance(source_value, bool):
            return float(source_value)
        if isinstance(source_value, str):
            try:
                return float(source_value.strip())
            except ValueError:
                pass

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        low = float(minimum) if isinstance(minimum, (int, float)) else 1.0
        high = float(maximum) if isinstance(maximum, (int, float)) else (low + 500.0)
        if high < low:
            high = low
        return round(rng.uniform(low, high), 2)

    def _infer_exact_identifier(
        self,
        field_name: str | None,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
    ) -> str | None:
        if not field_name:
            return None

        from_request = self._lookup_mapping_value(request_body, field_name)
        if from_request is not None:
            return str(from_request)
        from_path = self._lookup_mapping_value(path_parameters, field_name)
        if from_path is not None:
            return str(from_path)

        return None

    @staticmethod
    def _infer_any_identifier(
        self,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
    ) -> str | None:
        for source in (path_parameters, request_body):
            for key, value in source.items():
                if value is None:
                    continue
                if self._is_identifier_like_field(
                    field_name=str(key),
                    schema=None,
                    request_body=request_body,
                    path_parameters=path_parameters,
                ):
                    return str(value)
        return None

    @classmethod
    def _lookup_mapping_value(cls, mapping: dict[str, Any], field_name: str | None) -> Any | None:
        if not field_name:
            return None
        if field_name in mapping and mapping[field_name] is not None:
            return mapping[field_name]

        lowered = field_name.lower()
        for key, value in mapping.items():
            if str(key).lower() == lowered and value is not None:
                return value

        normalized = cls._normalize_key(field_name)
        for key, value in mapping.items():
            if cls._normalize_key(str(key)) == normalized and value is not None:
                return value

        return None

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @staticmethod
    def _path_context_hint(path_parameters: dict[str, Any]) -> str:
        ordered = [f"{key}={value}" for key, value in path_parameters.items()]
        return ", ".join(ordered) if ordered else "request"

    @staticmethod
    def _humanize_field(field_name: str | None) -> str:
        if not field_name:
            return "value"
        return re.sub(r"_+", " ", field_name).strip().lower()

    @staticmethod
    def _is_identifier_like_field(
        field_name: str | None,
        schema: dict[str, Any] | None,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
    ) -> bool:
        if not field_name:
            return False

        if isinstance(schema, dict):
            if schema.get("x-identifier") is True:
                return True
            if schema.get("x-primary-key") is True:
                return True
            fmt = str(schema.get("format", "")).lower()
            if fmt in {"uuid"}:
                return True

        context_keys = list(path_parameters.keys()) + list(request_body.keys())
        return IdentifierSemanticAnalyzer.is_identifier_like(field_name, context_keys=context_keys)

    def _ensure_path_context_in_text(self, text: str, path_parameters: dict[str, Any]) -> str:
        marker = self._path_context_hint(path_parameters)
        lowered_text = text.lower()
        has_any_path_value = any(str(value).lower() in lowered_text for value in path_parameters.values())
        if has_any_path_value:
            return text
        return f"{text} [{marker}]"

    @staticmethod
    def _generate_identifier(field_name: str | None, collection: str | None, rng: random.Random) -> str:
        base = SchemaSynthesizer._normalize_key(field_name or collection or "id")
        if not base:
            base = "id"
        cleaned = re.sub(r"(identifier|number|index|key|code|ref|token|uuid|id)$", "", base).strip()
        prefix = cleaned[:3] if len(cleaned) >= 3 else "id"
        return f"{prefix}_{rng.randint(1000, 9999)}"

    @staticmethod
    def _safe_slug(value: Any) -> str:
        text = str(value or "user").strip().lower()
        text = re.sub(r"[^a-z0-9]+", ".", text).strip(".")
        return text or "user"

    @staticmethod
    def _resource_title(collection: str | None) -> str:
        if not collection:
            return "Resource"
        name = collection.strip().replace("_", " ")
        if name.endswith("s") and len(name) > 1:
            name = name[:-1]
        name = name[:1].upper() + name[1:]
        return name
