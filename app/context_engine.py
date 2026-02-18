from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any


class ContextEngine:
    def __init__(self, history_limit: int = 200) -> None:
        self.history_limit = max(10, history_limit)
        self.context_store: dict[str, Any] = {"entities": {}, "history": []}
        self._lock = Lock()

    def clear(self) -> None:
        with self._lock:
            self.context_store = {"entities": {}, "history": []}

    def get_context_for_prompt(
        self,
        path_template: str,
        method: str,
        path_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        collection = self._collection_name(path_template)
        entities = self.context_store["entities"].get(collection, {})
        relevant_entity = self._resolve_entity_from_path(path_parameters or {}, entities)
        recent_history = self.context_store["history"][-10:]

        return {
            "collection": collection,
            "method": method.upper(),
            "path_parameters": deepcopy(path_parameters or {}),
            "known_entities_for_collection": deepcopy(entities),
            "resolved_entity": deepcopy(relevant_entity),
            "recent_history": deepcopy(recent_history),
        }

    def get_cached_get_response(
        self,
        path_template: str,
        path_parameters: dict[str, Any] | None = None,
    ) -> Any | None:
        collection = self._collection_name(path_template)
        entities = self.context_store["entities"].get(collection, {})
        if not entities:
            return None

        target_entity = self._resolve_entity_from_path(path_parameters or {}, entities)
        if target_entity is not None:
            return deepcopy(target_entity)

        # No resource identifier in path; return all known entities for the collection.
        return deepcopy(list(entities.values()))

    def update_from_response(
        self,
        method: str,
        path_template: str,
        path_parameters: dict[str, Any] | None,
        response_payload: Any,
    ) -> None:
        normalized_method = method.lower()
        collection = self._collection_name(path_template)
        path_parameters = path_parameters or {}

        with self._lock:
            if normalized_method in {"post", "put", "patch"}:
                entity_id = self._extract_entity_id(response_payload) or self._extract_path_identifier(path_parameters)
                if entity_id is not None and isinstance(response_payload, dict):
                    self.context_store["entities"].setdefault(collection, {})
                    self.context_store["entities"][collection][entity_id] = deepcopy(response_payload)
            elif normalized_method == "delete":
                entity_id = self._extract_path_identifier(path_parameters)
                if entity_id is not None:
                    self.context_store["entities"].setdefault(collection, {})
                    self.context_store["entities"][collection].pop(entity_id, None)

    def append_history(
        self,
        method: str,
        path_template: str,
        request_body: Any,
        response_payload: Any,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method.upper(),
            "path": path_template,
            "request_body": deepcopy(request_body),
            "response": deepcopy(response_payload),
        }
        with self._lock:
            self.context_store["history"].append(event)
            if len(self.context_store["history"]) > self.history_limit:
                self.context_store["history"] = self.context_store["history"][-self.history_limit :]

    @staticmethod
    def _collection_name(path_template: str) -> str:
        segments = [segment for segment in path_template.split("/") if segment]
        for segment in segments:
            if not segment.startswith("{"):
                return segment
        return "root"

    @staticmethod
    def _extract_entity_id(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        direct_keys = ("id", "uuid", "_id")
        for key in direct_keys:
            if key in payload and payload[key] is not None:
                return str(payload[key])
        for key, value in payload.items():
            if key.lower().endswith("id") and value is not None:
                return str(value)
        return None

    @staticmethod
    def _extract_path_identifier(path_parameters: dict[str, Any]) -> str | None:
        if not path_parameters:
            return None
        first_value = next(iter(path_parameters.values()))
        return str(first_value) if first_value is not None else None

    @staticmethod
    def _resolve_entity_from_path(path_parameters: dict[str, Any], entities: dict[str, Any]) -> Any | None:
        if not path_parameters:
            return None
        for value in path_parameters.values():
            key = str(value)
            if key in entities:
                return entities[key]
        return None

