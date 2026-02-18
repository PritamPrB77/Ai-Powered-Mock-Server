from __future__ import annotations

import random
import re
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.utils import json_dumps


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

            selected_source = source_obj.get(key)
            if selected_source is None and key in request_body:
                selected_source = request_body[key]
            if selected_source is None and key in path_parameters:
                selected_source = path_parameters[key]

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

        if field_name and field_name in request_body and request_body[field_name] is not None:
            return str(request_body[field_name])
        if field_name and field_name in path_parameters and path_parameters[field_name] is not None:
            return str(path_parameters[field_name])

        if field.endswith("id") or field == "id":
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

        min_len = schema.get("minLength")
        base = f"value_{rng.randint(100, 999)}"
        if isinstance(min_len, int) and min_len > len(base):
            base = base + ("x" * (min_len - len(base)))
        return base

    @staticmethod
    def _build_integer(schema: dict[str, Any], source_value: Any, rng: random.Random) -> int:
        if isinstance(source_value, int) and not isinstance(source_value, bool):
            return source_value

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

        if field_name in request_body and request_body[field_name] is not None:
            return str(request_body[field_name])
        if field_name in path_parameters and path_parameters[field_name] is not None:
            return str(path_parameters[field_name])

        lowered = field_name.lower()
        for key, value in request_body.items():
            if str(key).lower() == lowered and value is not None:
                return str(value)
        for key, value in path_parameters.items():
            if str(key).lower() == lowered and value is not None:
                return str(value)

        return None

    @staticmethod
    def _infer_any_identifier(
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
    ) -> str | None:
        for key, value in path_parameters.items():
            if str(key).lower().endswith("id") and value is not None:
                return str(value)
        for key, value in request_body.items():
            if str(key).lower().endswith("id") and value is not None:
                return str(value)
        return None

    @staticmethod
    def _generate_identifier(field_name: str | None, collection: str | None, rng: random.Random) -> str:
        base = (field_name or collection or "id").lower()
        if "order" in base:
            prefix = "ord"
        elif "customer" in base:
            prefix = "cus"
        elif "ticket" in base:
            prefix = "tkt"
        elif "catalog" in base or "sku" in base:
            prefix = "sku"
        else:
            prefix = "id"
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
