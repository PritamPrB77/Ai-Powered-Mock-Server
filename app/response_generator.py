from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi.responses import JSONResponse
from openapi_core import OpenAPI
from openapi_core.contrib.starlette.responses import StarletteOpenAPIResponse

from app.config import Settings
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
                "No generation source available. Enable Seq2Seq or configure OPENROUTER_API_KEY."
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
                        temperature=temperature,
                    )
                    if candidate is None:
                        last_error = f"{source} source unavailable."
                        continue

                    candidate = self.schema_synthesizer.build(
                        schema=operation.response_schema,
                        request_body=validated_request_body,
                        path_parameters=context_payload.get("path_parameters"),
                        collection=str(context_payload.get("collection", "")),
                        source_candidate=candidate,
                    )

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

        schema_candidate = self._build_schema_fallback_candidate(
            operation=operation,
            validated_request_body=validated_request_body,
            context_payload=context_payload,
        )
        if schema_candidate is not None and self._is_structurally_valid(
            operation=operation,
            openapi=openapi,
            openapi_request=openapi_request,
            candidate=schema_candidate,
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
        temperature: float | None = None,
    ) -> Any | None:
        if source == "seq2seq":
            if self.seq2seq_generator is None:
                return None
            return await self.seq2seq_generator.generate_json(prompt, temperature=temperature)
        if source == "openrouter":
            return await self._call_openrouter(prompt, temperature=temperature)
        raise ResponseGenerationError(f"Unknown generation source: {source}")

    def _generation_source_order(self) -> list[str]:
        sources: list[str] = []
        has_seq2seq = self.seq2seq_generator is not None and self.seq2seq_generator.enabled
        if has_seq2seq:
            sources.append("seq2seq")

        openrouter_allowed = self.settings.openrouter_enabled and bool(self.settings.openrouter_api_key)
        if openrouter_allowed and (not has_seq2seq or self.settings.openrouter_fallback_enabled):
            sources.append("openrouter")
        return sources

    def _has_active_generation_source(self) -> bool:
        return bool(self._generation_source_order())

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

    async def _call_openrouter(self, prompt: str, temperature: float | None = None) -> Any:
        selected_temperature = (
            self.settings.generation_temperature
            if temperature is None
            else float(temperature)
        )
        payload = {
            "model": self.settings.openrouter_model,
            "temperature": selected_temperature,
            "max_tokens": self.settings.generation_max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an API response generator. "
                        "Return only valid JSON. Do not include markdown or explanations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.openrouter_http_referer,
            "X-Title": self.settings.openrouter_app_title,
        }

        timeout = httpx.Timeout(self.settings.generation_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self.settings.openrouter_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            llm_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ResponseGenerationError(f"Unexpected OpenRouter response payload: {data}") from exc

        return extract_json_value(llm_text)

    @staticmethod
    def _build_prompt(
        operation: OperationMeta,
        validated_request_body: Any,
        context_payload: dict[str, Any],
    ) -> str:
        return (
            "You are generating a dynamic mock API response.\n\n"
            f"Endpoint: {operation.path}\n"
            f"Method: {operation.method.upper()}\n\n"
            "Sequence-to-sequence context:\n"
            "1) Read request schema and validated request data.\n"
            "2) Map request intent into response schema fields.\n"
            "3) Produce final JSON output.\n\n"
            f"Request Schema:\n{json_dumps(operation.request_schema)}\n\n"
            f"Response Schema:\n{json_dumps(operation.response_schema)}\n\n"
            f"Validated Request Body:\n{json_dumps(validated_request_body)}\n\n"
            f"System Context:\n{json_dumps(context_payload)}\n\n"
            "Output constraints:\n"
            "- Return JSON only.\n"
            "- Match response schema exactly.\n"
            "- Do not add extra fields.\n"
            "- Ensure logical consistency with request and context."
        )
