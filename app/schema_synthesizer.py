from __future__ import annotations

import random
import re
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

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
    IDENTIFIER_POSITIVE_STEMS = (
        "id",
        "identifier",
        "key",
        "code",
        "number",
        "index",
        "sequence",
        "serial",
        "ordinal",
        "position",
        "rank",
        "level",
        "reference",
        "ref",
        "token",
        "uuid",
        "slug",
        "handle",
        "parent",
        "child",
    )
    IDENTIFIER_NEGATIVE_STEMS = (
        "name",
        "title",
        "summary",
        "description",
        "message",
        "content",
        "text",
        "email",
        "address",
        "status",
        "note",
    )
    GENERIC_PLACEHOLDER_TEXTS = (
        "john doe",
        "jane doe",
        "john smith",
        "jane smith",
        "lorem ipsum",
        "foo bar",
        "sample text",
        "test user",
        "customer name",
    )

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
        for key, prop_schema in props.items():
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
        request_list = self._lookup_mapping_value(request_body, field_name) if field_name else None
        if not source_list and isinstance(request_list, list):
            source_list = request_list

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
            if not self._is_generic_placeholder_text(source_value, field_name, schema):
                if self._should_bind_path_context(
                    field_name=field_name,
                    schema=schema,
                    source_value=source_value,
                    request_body=request_body,
                    path_parameters=path_parameters,
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
            best_seed = self._best_text_seed(request_body)
            suffix = f".{rng.randint(10, 99)}"
            local_name = self._safe_slug(
                best_seed
                or f"user{generic_id or rng.randint(100, 999)}{suffix}"
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
            return f"{rng.choice(self.FIRST_NAMES)} {rng.choice(self.LAST_NAMES)}"

        if self._should_generate_contextual_text(
            field_name=field_name,
            schema=schema,
            request_body=request_body,
            path_parameters=path_parameters,
        ):
            text = self._build_contextual_text_value(
                field_name=field_name,
                collection=collection,
                request_body=request_body,
                path_parameters=path_parameters,
                rng=rng,
            )
            min_len = schema.get("minLength")
            if isinstance(min_len, int) and min_len > len(text):
                text = text + ("x" * (min_len - len(text)))
            return text

        min_len = schema.get("minLength")
        base_key = self._normalize_key(field_name or "value")
        base = f"{base_key}_{rng.randint(100, 999)}"
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

    def _build_contextual_text_value(
        self,
        field_name: str | None,
        collection: str | None,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
        rng: random.Random,
    ) -> str:
        pieces: list[str] = []
        descriptor = self._humanize_field(field_name)
        pieces.append(descriptor if descriptor != "value" else "detail")

        if collection:
            pieces.append(f"for {self._resource_title(collection)}")
        if path_parameters:
            pieces.append(f"({self._path_context_hint(path_parameters)})")

        seed = self._best_text_seed(request_body)
        if seed:
            compact_seed = " ".join(seed.split())
            if len(compact_seed) > 48:
                compact_seed = compact_seed[:45].rstrip() + "..."
            pieces.append(f"based on {compact_seed}")

        pieces.append(f"ref-{rng.randint(1000, 9999)}")
        return " ".join(pieces)

    def _should_generate_contextual_text(
        self,
        field_name: str | None,
        schema: dict[str, Any],
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
    ) -> bool:
        return self._is_descriptive_string_field(
            field_name=field_name,
            schema=schema,
            request_body=request_body,
            path_parameters=path_parameters,
        )

    def _should_bind_path_context(
        self,
        field_name: str | None,
        schema: dict[str, Any],
        source_value: str,
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
    ) -> bool:
        if not path_parameters:
            return False
        if not source_value.strip():
            return False
        if not self._is_descriptive_string_field(
            field_name=field_name,
            schema=schema,
            request_body=request_body,
            path_parameters=path_parameters,
        ):
            return False
        lowered_text = source_value.lower()
        return not any(
            str(value).lower() in lowered_text
            for value in path_parameters.values()
            if value is not None
        )

    def _is_descriptive_string_field(
        self,
        field_name: str | None,
        schema: dict[str, Any],
        request_body: dict[str, Any],
        path_parameters: dict[str, Any],
    ) -> bool:
        if not field_name:
            return False
        if self._is_identifier_like_field(
            field_name=field_name,
            schema=schema,
            request_body=request_body,
            path_parameters=path_parameters,
        ):
            return False

        field_text = field_name.lower()
        if "name" in field_text:
            return False

        fmt = str(schema.get("format", "")).lower()
        if fmt in {"date", "date-time", "email", "uuid", "uri", "url", "ipv4", "ipv6", "hostname"}:
            return False

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return False
        return True

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
            if schema.get("x-identifier") is True or schema.get("x-primary-key") is True:
                return True
            fmt = str(schema.get("format", "")).lower()
            if fmt == "uuid":
                return True

        tokens = SchemaSynthesizer._tokenize_field_name(field_name)
        if not tokens:
            return False

        positive_score = 0.0
        negative_score = 0.0
        for token in tokens:
            positive_similarity = max(
                SchemaSynthesizer._token_similarity(token, stem)
                for stem in SchemaSynthesizer.IDENTIFIER_POSITIVE_STEMS
            )
            negative_similarity = max(
                SchemaSynthesizer._token_similarity(token, stem)
                for stem in SchemaSynthesizer.IDENTIFIER_NEGATIVE_STEMS
            )
            if positive_similarity >= 0.9:
                positive_score += 1.0
            else:
                positive_score += positive_similarity * 0.5
            if negative_similarity >= 0.9:
                negative_score += 1.0
            else:
                negative_score += negative_similarity * 0.5

        compact_field = SchemaSynthesizer._normalize_key(field_name)
        context_keys = list(path_parameters.keys()) + list(request_body.keys())
        for key in context_keys:
            normalized_key = SchemaSynthesizer._normalize_key(str(key))
            if not normalized_key:
                continue
            if compact_field == normalized_key:
                positive_score += 0.4
            if compact_field.endswith(normalized_key) or normalized_key.endswith(compact_field):
                positive_score += 0.2

        if compact_field.isdigit():
            positive_score += 0.2
        if any(char.isdigit() for char in compact_field):
            positive_score += 0.1

        return positive_score > (negative_score + 0.25)

    @staticmethod
    def _tokenize_field_name(value: str | None) -> list[str]:
        text = str(value or "")
        if not text:
            return []
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
        tokens = [token for token in text.split() if token]
        return [SchemaSynthesizer._canonicalize_token(token) for token in tokens if token]

    @staticmethod
    def _canonicalize_token(token: str) -> str:
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    @staticmethod
    def _token_similarity(token: str, reference: str) -> float:
        if not token or not reference:
            return 0.0
        if token == reference:
            return 1.0
        if token.startswith(reference) or reference.startswith(token):
            return 0.92
        longest = 0
        token_len = len(token)
        ref_len = len(reference)
        for i in range(token_len):
            for j in range(i + 1, token_len + 1):
                fragment = token[i:j]
                if fragment and fragment in reference:
                    longest = max(longest, len(fragment))
        return float(longest) / float(max(token_len, ref_len))

    def _best_text_seed(self, request_body: dict[str, Any]) -> str | None:
        if not isinstance(request_body, dict):
            return None

        for value in request_body.values():
            if isinstance(value, str):
                text = value.strip()
                if 2 <= len(text) <= 80:
                    return text
        return None

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
        prefix = base[:6] if len(base) >= 2 else "id"
        return f"{prefix}_{rng.randint(1000, 9999)}"

    @staticmethod
    def _safe_slug(value: Any) -> str:
        text = str(value or "user").strip().lower()
        text = re.sub(r"[^a-z0-9]+", ".", text).strip(".")
        return text or "user"

    @staticmethod
    def _is_generic_placeholder_text(
        value: str,
        field_name: str | None,
        schema: dict[str, Any] | None = None,
    ) -> bool:
        text = value.strip().lower()
        if not text:
            return False
        if text in SchemaSynthesizer.GENERIC_PLACEHOLDER_TEXTS:
            return True
        if "john doe" in text or "jane doe" in text:
            return True
        if "lorem ipsum" in text:
            return True
        if "customer@example.com" in text:
            return True

        field = (field_name or "").lower()
        fmt = str((schema or {}).get("format", "")).lower()
        if (fmt == "email" or "email" in field) and text.endswith("@example.com"):
            return True
        return False

    @staticmethod
    def _resource_title(collection: str | None) -> str:
        if not collection:
            return "Resource"
        name = collection.strip().replace("_", " ")
        if name.endswith("s") and len(name) > 1:
            name = name[:-1]
        name = name[:1].upper() + name[1:]
        return name
