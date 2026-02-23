from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from fastapi.responses import JSONResponse
from openapi_core import OpenAPI
from openapi_core.contrib.starlette.responses import StarletteOpenAPIResponse

from app.config import Settings
from app.field_semantics import IdentifierSemanticAnalyzer
from app.openapi_loader import OperationMeta
from app.schema_synthesizer import SchemaSynthesizer
from app.seq2seq_generator import Seq2SeqGenerator
from app.semantic_validator import SemanticValidator
from app.utils import extract_json_value, json_dumps


logger = logging.getLogger(__name__)


class ResponseGenerationError(Exception):
    pass


class DynamicResponseGenerator:
    def __init__(
        self,
        settings: Settings,
        semantic_validator: SemanticValidator,
        seq2seq_generator: Seq2SeqGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.semantic_validator = semantic_validator
        self.seq2seq_generator = seq2seq_generator
        self.schema_synthesizer = SchemaSynthesizer()

    async def generate(
        self,
        operation: OperationMeta,
        openapi: OpenAPI,
        openapi_request: Any,
        validated_request_body: Any,
        context_payload: dict[str, Any],
        n: int,
        temperature: float | None = None,
    ) -> list[Any]:
        if not self._has_active_generation_source():
            raise ResponseGenerationError(
                "No generation source available. Enable Seq2Seq, or configure the selected "
                "GENERATION_PROVIDER (ollama/openrouter)."
            )

        output: list[Any] = []
        for _ in range(n):
            candidate = await self._generate_single_with_retries(
                operation=operation,
                openapi=openapi,
                openapi_request=openapi_request,
                validated_request_body=validated_request_body,
                context_payload=context_payload,
                temperature=temperature,
            )
            output.append(candidate)
        return output

    async def _generate_single_with_retries(
        self,
        operation: OperationMeta,
        openapi: OpenAPI,
        openapi_request: Any,
        validated_request_body: Any,
        context_payload: dict[str, Any],
        temperature: float | None,
    ) -> Any:
        last_error = "No candidate generated."
        fallback_candidate: Any | None = None
        fallback_source = ""
        best_semantic_score = -1.0
        semantic_schema_source = {
            "endpoint": operation.path,
            "method": operation.method.upper(),
            "response_schema": operation.response_schema,
        }
        semantic_context_source = {
            "request_body": validated_request_body,
            "path_parameters": context_payload.get("path_parameters"),
            "query_parameters": context_payload.get("query_parameters"),
            "api_info": context_payload.get("api_info"),
            "operation_summary": context_payload.get("operation_summary"),
            "operation_description": context_payload.get("operation_description"),
            "resolved_entity": context_payload.get("resolved_entity"),
            "collection": context_payload.get("collection"),
        }

        for attempt in range(1, self.settings.max_retry_attempts + 1):
            prompt = self._build_prompt(
                operation=operation,
                validated_request_body=validated_request_body,
                context_payload=context_payload,
            )
            for source in self._generation_source_order():
                try:
                    candidate = await self._generate_candidate(
                        source=source,
                        prompt=prompt,
                        context_payload=context_payload,
                        temperature=temperature,
                    )
                    if candidate is None:
                        last_error = f"{source} source unavailable."
                        continue

                    candidate = self._normalize_candidate_shape(operation.response_schema, candidate)
                    if not self._matches_response_shape(operation.response_schema, candidate):
                        last_error = f"{source} candidate JSON shape does not match response schema type."
                        continue

                    should_use_raw_candidate = (
                        source == "openrouter"
                        and self.settings.openrouter_strict_generation
                    )
                    if not should_use_raw_candidate:
                        candidate = self.schema_synthesizer.build(
                            schema=operation.response_schema,
                            request_body=validated_request_body,
                            path_parameters=context_payload.get("path_parameters"),
                            collection=str(context_payload.get("collection", "")),
                            source_candidate=candidate,
                        )

                    candidate = self._enforce_identifier_consistency(candidate, context_payload)
                    if self._has_context_variation_conflict(candidate, context_payload):
                        last_error = f"{source} candidate repeated content from prior context for different identifiers."
                        continue

                    if not self._is_structurally_valid(
                        operation=operation,
                        openapi=openapi,
                        openapi_request=openapi_request,
                        candidate=candidate,
                    ):
                        last_error = f"{source} candidate failed OpenAPI response validation."
                        continue

                    semantic_score = 1.0
                    if self.settings.semantic_validation_enabled:
                        semantic_schema_score = await self.semantic_validator.score(
                            semantic_schema_source,
                            candidate,
                        )
                        semantic_context_score = await self.semantic_validator.score(
                            semantic_context_source,
                            candidate,
                        )
                        semantic_score = (0.75 * semantic_schema_score) + (0.25 * semantic_context_score)
                    if semantic_score > best_semantic_score:
                        best_semantic_score = semantic_score
                        fallback_candidate = candidate
                        fallback_source = source

                    if (
                        self.settings.semantic_validation_enabled
                        and semantic_score < self.settings.semantic_similarity_threshold
                    ):
                        last_error = (
                            f"{source} candidate failed semantic similarity validation "
                            f"(score={semantic_score:.3f}, "
                            f"threshold={self.settings.semantic_similarity_threshold:.3f})."
                        )
                        continue

                    return candidate
                except Exception as exc:  # noqa: BLE001
                    last_error = f"{source} generation failed: {exc}"
                    logger.debug(
                        "Response generation attempt %s via %s failed: %s",
                        attempt,
                        source,
                        exc,
                    )

        if fallback_candidate is not None and self.settings.semantic_fail_open:
            logger.warning(
                "Returning structurally valid fallback candidate after semantic retries failed. "
                "source=%s best_score=%.3f threshold=%.3f",
                fallback_source,
                best_semantic_score,
                self.settings.semantic_similarity_threshold,
            )
            return fallback_candidate

        if not self._should_skip_schema_fallback():
            schema_candidate = self._build_schema_fallback_candidate(
                operation=operation,
                validated_request_body=validated_request_body,
                context_payload=context_payload,
            )
            if (
                schema_candidate is not None
                and self._is_structurally_valid(
                    operation=operation,
                    openapi=openapi,
                    openapi_request=openapi_request,
                    candidate=schema_candidate,
                )
            ):
                logger.warning(
                    "Returning schema-derived fallback candidate after generation retries failed."
                )
                return schema_candidate

        raise ResponseGenerationError(
            f"Failed to generate a valid response after {self.settings.max_retry_attempts} attempts. "
            f"Last error: {last_error}"
        )

    def _is_structurally_valid(
        self,
        operation: OperationMeta,
        openapi: OpenAPI,
        openapi_request: Any,
        candidate: Any,
    ) -> bool:
        if operation.response_schema is None:
            return True

        response = JSONResponse(
            status_code=operation.response_status_code,
            content=candidate,
        )
        openapi_response = StarletteOpenAPIResponse(response)
        result = openapi.unmarshal_response(openapi_request, openapi_response)
        return len(result.errors) == 0

    async def _generate_candidate(
        self,
        source: str,
        prompt: str,
        context_payload: dict[str, Any],
        temperature: float | None = None,
    ) -> Any | None:
        if source == "seq2seq":
            if self.seq2seq_generator is None:
                return None
            return await self.seq2seq_generator.generate_json(prompt, temperature=temperature)
        if source == "ollama":
            return await self._call_ollama(prompt, temperature=temperature)
        if source == "openrouter":
            return await self._call_openrouter(
                prompt=prompt,
                context_payload=context_payload,
                temperature=temperature,
            )
        raise ResponseGenerationError(f"Unknown generation source: {source}")

    def _generation_source_order(self) -> list[str]:
        sources: list[str] = []
        has_seq2seq = self.seq2seq_generator is not None and self.seq2seq_generator.enabled

        provider = str(self.settings.generation_provider).strip().lower()
        if provider not in {"ollama", "openrouter"}:
            logger.warning(
                "Unknown GENERATION_PROVIDER '%s'. Falling back to 'openrouter'.",
                self.settings.generation_provider,
            )
            provider = "openrouter"

        ollama_allowed = (
            self.settings.ollama_enabled
            and bool(self.settings.ollama_url)
            and bool(self.settings.ollama_model)
        )
        openrouter_allowed = self.settings.openrouter_enabled and bool(self.settings.openrouter_api_key)

        if provider == "openrouter" and self.settings.openrouter_strict_generation:
            if openrouter_allowed:
                return ["openrouter"]
            return []

        if provider == "ollama":
            if ollama_allowed:
                sources.append("ollama")
        elif provider == "openrouter":
            if openrouter_allowed:
                sources.append("openrouter")

        if has_seq2seq and (
            not sources
            or (
                provider == "ollama"
                and self.settings.ollama_fallback_enabled
            )
            or (
                provider == "openrouter"
                and self.settings.openrouter_fallback_enabled
            )
        ):
            sources.append("seq2seq")

        if provider == "ollama" and openrouter_allowed and self.settings.openrouter_fallback_enabled:
            if "openrouter" not in sources:
                sources.append("openrouter")
        if provider == "openrouter" and ollama_allowed and self.settings.ollama_fallback_enabled:
            if "ollama" not in sources:
                sources.append("ollama")
        return sources

    def _has_active_generation_source(self) -> bool:
        return bool(self._generation_source_order())

    def _should_skip_schema_fallback(self) -> bool:
        return self._provider_name() == "openrouter" and self.settings.openrouter_strict_generation

    def _build_schema_fallback_candidate(
        self,
        operation: OperationMeta,
        validated_request_body: Any,
        context_payload: dict[str, Any],
    ) -> Any | None:
        return self.schema_synthesizer.build(
            schema=operation.response_schema,
            request_body=validated_request_body,
            path_parameters=context_payload.get("path_parameters"),
            collection=str(context_payload.get("collection", "")),
            source_candidate=None,
        )

    async def _call_openrouter(
        self,
        prompt: str,
        context_payload: dict[str, Any],
        temperature: float | None = None,
    ) -> Any:
        selected_temperature = (
            self.settings.generation_temperature
            if temperature is None
            else float(temperature)
        )
        payload = {
            "model": self.settings.openrouter_model,
            "temperature": selected_temperature,
            "max_tokens": self.settings.generation_max_tokens,
            "messages": self._build_openrouter_messages(prompt=prompt, context_payload=context_payload),
            "provider": {
                "allow_fallbacks": self.settings.openrouter_allow_provider_fallbacks,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_http_referer,
            "X-Title": self.settings.openrouter_app_title,
        }

        timeout = httpx.Timeout(self.settings.generation_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    self.settings.openrouter_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = self._format_openrouter_http_error(exc)
                raise ResponseGenerationError(f"OpenRouter request failed: {detail}") from exc
            except httpx.RequestError as exc:
                raise ResponseGenerationError(f"OpenRouter connection failed: {exc}") from exc
            data = response.json()

        try:
            llm_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ResponseGenerationError(f"Unexpected OpenRouter response payload: {data}") from exc

        return extract_json_value(llm_text)

    def _build_openrouter_messages(
        self,
        prompt: str,
        context_payload: dict[str, Any],
    ) -> list[dict[str, str]]:
        system_prompt = self.settings.openrouter_system_prompt.strip()
        if not system_prompt:
            system_prompt = (
                "You are an intelligent API response generator inside a dynamic mock server.\n"
                "Output strictly valid JSON and do not add extra fields."
            )

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        memory = self._build_openrouter_memory(context_payload)
        if memory is not None:
            messages.append(
                {
                    "role": "system",
                    "content": f"Runtime context memory:\n{json_dumps(memory)}",
                }
            )

        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_openrouter_memory(self, context_payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(context_payload, dict):
            return None

        known_entities = context_payload.get("known_entities_for_collection")
        recent_history = context_payload.get("recent_history")
        selected_history: list[dict[str, Any]] = []

        if isinstance(recent_history, list):
            history_limit = self.settings.openrouter_history_turns
            if history_limit > 0:
                for event in recent_history[-history_limit:]:
                    if not isinstance(event, dict):
                        continue
                    selected_history.append(
                        {
                            "method": event.get("method"),
                            "path": event.get("path"),
                            "request_body": event.get("request_body"),
                            "response": event.get("response"),
                        }
                    )

        return {
            "collection": context_payload.get("collection"),
            "method": context_payload.get("method"),
            "path_parameters": context_payload.get("path_parameters"),
            "query_parameters": context_payload.get("query_parameters"),
            "resolved_entity": context_payload.get("resolved_entity"),
            "known_entity_count": len(known_entities) if isinstance(known_entities, dict) else 0,
            "recent_history": selected_history,
        }

    def _provider_name(self) -> str:
        provider = str(self.settings.generation_provider).strip().lower()
        if provider in {"ollama", "openrouter"}:
            return provider
        return "openrouter"

    @classmethod
    def _enforce_identifier_consistency(
        cls,
        candidate: Any,
        context_payload: dict[str, Any],
    ) -> Any:
        if not isinstance(candidate, (dict, list)):
            return candidate

        path_parameters = context_payload.get("path_parameters")
        query_parameters = context_payload.get("query_parameters")
        identifier_map = cls._build_identifier_map(path_parameters, query_parameters)
        if not identifier_map:
            return candidate

        return cls._apply_identifier_map(candidate, identifier_map)

    @classmethod
    def _build_identifier_map(
        cls,
        path_parameters: Any,
        query_parameters: Any,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for source in (path_parameters, query_parameters):
            if not isinstance(source, dict):
                continue
            for raw_key, raw_value in source.items():
                if raw_value is None:
                    continue
                key = str(raw_key)
                normalized = cls._normalize_key(key)
                if not normalized:
                    continue
                merged[normalized] = raw_value

        if not merged:
            return {}

        expanded: dict[str, Any] = dict(merged)
        for key, value in list(merged.items()):
            if key.endswith("number") and len(key) > len("number"):
                expanded[key[:-6]] = value
            if key.endswith("num") and len(key) > len("num"):
                expanded[key[:-3]] = value
            if key.endswith("id") and len(key) > len("id"):
                expanded[key[:-2]] = value
        return expanded

    @classmethod
    def _apply_identifier_map(cls, node: Any, identifier_map: dict[str, Any]) -> Any:
        if isinstance(node, list):
            return [cls._apply_identifier_map(item, identifier_map) for item in node]
        if not isinstance(node, dict):
            return node

        updated: dict[str, Any] = {}
        for key, value in node.items():
            if cls._is_identifier_like_field(key, context_keys=identifier_map.keys()):
                replacement = cls._lookup_identifier_value(key, identifier_map)
                if replacement is not None:
                    updated[key] = cls._coerce_like(value, replacement)
                    continue
            updated[key] = cls._apply_identifier_map(value, identifier_map)
        return updated

    @classmethod
    def _lookup_identifier_value(cls, field_name: str, identifier_map: dict[str, Any]) -> Any | None:
        normalized = cls._normalize_key(field_name)
        if not normalized:
            return None
        if normalized in identifier_map:
            return identifier_map[normalized]

        for candidate_key, candidate_value in identifier_map.items():
            if normalized.endswith(candidate_key) or candidate_key.endswith(normalized):
                return candidate_value
        return None

    @staticmethod
    def _coerce_like(existing_value: Any, replacement: Any) -> Any:
        if isinstance(existing_value, bool):
            return existing_value
        if isinstance(existing_value, int):
            try:
                return int(replacement)
            except (TypeError, ValueError):
                return existing_value
        if isinstance(existing_value, float):
            try:
                return float(replacement)
            except (TypeError, ValueError):
                return existing_value
        return replacement

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    @staticmethod
    def _is_identifier_like_field(
        field_name: str,
        context_keys: Any = None,
    ) -> bool:
        keys = list(context_keys) if context_keys is not None else []
        return IdentifierSemanticAnalyzer.is_identifier_like(field_name, context_keys=keys)

    @classmethod
    def _has_context_variation_conflict(
        cls,
        candidate: Any,
        context_payload: dict[str, Any],
    ) -> bool:
        if not isinstance(candidate, dict):
            return False

        candidate_signature = cls._content_signature(candidate)
        if not candidate_signature:
            return False

        recent_history = context_payload.get("recent_history")
        if not isinstance(recent_history, list) or not recent_history:
            return False

        candidate_identifiers = cls._extract_identifiers(candidate)
        if not candidate_identifiers:
            return False

        for event in recent_history[-5:]:
            if not isinstance(event, dict):
                continue
            prior_response = event.get("response")
            if not isinstance(prior_response, dict):
                continue
            prior_identifiers = cls._extract_identifiers(prior_response)
            if not prior_identifiers:
                continue
            if prior_identifiers == candidate_identifiers:
                continue
            prior_signature = cls._content_signature(prior_response)
            if prior_signature and prior_signature == candidate_signature:
                return True
        return False

    @staticmethod
    def _extract_identifiers(payload: dict[str, Any]) -> dict[str, str]:
        identifiers: dict[str, str] = {}
        for key, value in payload.items():
            key_text = str(key).lower()
            if value is None:
                continue
            if IdentifierSemanticAnalyzer.is_identifier_like(key_text):
                identifiers[key_text] = str(value)
        return identifiers

    @staticmethod
    def _content_signature(payload: dict[str, Any]) -> str:
        tokens: list[str] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            key_text = str(key).lower()
            if any(token in key_text for token in ("translation", "commentary", "sanskrit", "text", "content")):
                text = value.strip().lower()
                if text:
                    tokens.append(text)
        return "|".join(tokens)

    @staticmethod
    def _format_openrouter_http_error(exc: httpx.HTTPStatusError) -> str:
        response = exc.response
        if response is None:
            return str(exc)

        response_text = response.text.strip()
        try:
            payload = response.json()
        except ValueError:
            return response_text or str(exc)

        if not isinstance(payload, dict):
            return response_text or str(exc)

        error_obj = payload.get("error")
        if not isinstance(error_obj, dict):
            return response_text or str(exc)

        code = error_obj.get("code")
        message = error_obj.get("message")
        metadata = error_obj.get("metadata")

        provider_name = ""
        upstream_message = ""
        if isinstance(metadata, dict):
            provider_raw = metadata.get("provider_name")
            if isinstance(provider_raw, str):
                provider_name = provider_raw

            raw_payload = metadata.get("raw")
            if isinstance(raw_payload, str) and raw_payload.strip():
                try:
                    parsed_raw = json.loads(raw_payload)
                except ValueError:
                    parsed_raw = None
                if isinstance(parsed_raw, dict):
                    raw_error = parsed_raw.get("error")
                    if isinstance(raw_error, dict):
                        raw_message = raw_error.get("message")
                        if isinstance(raw_message, str):
                            upstream_message = raw_message

        final_message = upstream_message or (message if isinstance(message, str) else "")
        if not final_message:
            final_message = response_text or str(exc)

        provider_text = f"provider={provider_name}, " if provider_name else ""
        code_text = f"code={code}, " if code is not None else ""
        prefix = f"{provider_text}{code_text}".rstrip(", ")
        if prefix:
            return f"({prefix}) {final_message}"
        return final_message

    @staticmethod
    def _matches_response_shape(schema: dict[str, Any] | None, candidate: Any) -> bool:
        if not isinstance(schema, dict):
            return True
        expected_type = schema.get("type")
        if expected_type == "object":
            return isinstance(candidate, dict)
        if expected_type == "array":
            return isinstance(candidate, list)
        return True

    @staticmethod
    def _normalize_candidate_shape(schema: dict[str, Any] | None, candidate: Any) -> Any:
        if not isinstance(schema, dict):
            return candidate

        expected_type = schema.get("type")
        if expected_type == "array":
            if isinstance(candidate, list):
                return candidate
            if isinstance(candidate, dict):
                for key in ("items", "data", "results", "result", "chapters", "values", "response"):
                    value = candidate.get(key)
                    if isinstance(value, list):
                        return value
                # When model emits a single item object for an array schema,
                # wrap it to satisfy the top-level array requirement.
                return [candidate]
            return candidate

        if expected_type == "object":
            if isinstance(candidate, dict):
                return candidate
            if isinstance(candidate, list) and len(candidate) == 1 and isinstance(candidate[0], dict):
                return candidate[0]
            return candidate

        return candidate

    async def _call_ollama(self, prompt: str, temperature: float | None = None) -> Any:
        selected_temperature = (
            self.settings.generation_temperature
            if temperature is None
            else float(temperature)
        )
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": selected_temperature,
                "num_predict": self.settings.ollama_num_predict,
                "top_k": self.settings.ollama_top_k,
                "top_p": self.settings.ollama_top_p,
                "seed": int(time.time_ns() % 2147483647),
            },
        }

        timeout = httpx.Timeout(self.settings.generation_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    self.settings.ollama_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                response_text = exc.response.text.strip() if exc.response is not None else ""
                detail = response_text or str(exc)
                raise ResponseGenerationError(f"Ollama request failed: {detail}") from exc
            data = response.json()

        try:
            llm_text = data["response"]
        except (KeyError, TypeError) as exc:
            raise ResponseGenerationError(f"Unexpected Ollama response payload: {data}") from exc

        if not isinstance(llm_text, str):
            raise ResponseGenerationError(f"Unexpected Ollama response payload: {data}")

        return extract_json_value(llm_text)

    @staticmethod
    def _build_prompt(
        operation: OperationMeta,
        validated_request_body: Any,
        context_payload: dict[str, Any],
    ) -> str:
        generation_nonce = time.time_ns()
        path_parameters = context_payload.get("path_parameters")
        query_parameters = context_payload.get("query_parameters")
        api_info = context_payload.get("api_info")
        operation_summary = context_payload.get("operation_summary")
        operation_description = context_payload.get("operation_description")
        return (
            "Generate a dynamic API response.\n\n"
            f"API Info:\n{json_dumps(api_info)}\n\n"
            f"Endpoint: {operation.path}\n"
            f"Method: {operation.method.upper()}\n\n"
            f"Operation Summary:\n{json_dumps(operation_summary)}\n\n"
            f"Operation Description:\n{json_dumps(operation_description)}\n\n"
            "Context requirements:\n"
            "- Use request/response schemas strictly.\n"
            "- Keep semantic consistency across related fields.\n"
            "- Avoid placeholder-style text.\n"
            "- Keep variation natural across runs while preserving identifiers.\n\n"
            f"Request Schema:\n{json_dumps(operation.request_schema)}\n\n"
            f"Response Schema:\n{json_dumps(operation.response_schema)}\n\n"
            f"Validated Request Body:\n{json_dumps(validated_request_body)}\n\n"
            f"Path Parameters:\n{json_dumps(path_parameters)}\n\n"
            f"Query Parameters:\n{json_dumps(query_parameters)}\n\n"
            f"System Context:\n{json_dumps(context_payload)}\n\n"
            f"Generation Nonce:\n{generation_nonce}\n\n"
            "Output constraints:\n"
            "- Return JSON only.\n"
            "- Match response schema exactly.\n"
            "- Use field names exactly as in Response Schema.\n"
            "- Preserve identifier values from path parameters even when naming style differs "
            "(example: chapterNumber -> chapter_number).\n"
            "- For different path parameter values, vary content fields accordingly.\n"
            "- Preserve context continuity with known entities and recent history.\n"
            "- Keep identifiers stable but vary non-identifier fields realistically.\n"
            "- Do not add extra fields.\n"
            "- Ensure logical consistency with request and context."
        )
