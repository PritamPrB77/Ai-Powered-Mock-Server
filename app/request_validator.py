from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request
from openapi_core import OpenAPI
from openapi_core.contrib.starlette.requests import StarletteOpenAPIRequest


class RequestValidationError(Exception):
    def __init__(self, details: list[str]) -> None:
        self.details = details
        super().__init__("; ".join(details))


@dataclass(slots=True)
class RequestValidationResult:
    openapi_request: StarletteOpenAPIRequest
    body: Any
    parameters: Any
    security: Any


class OpenAPIRequestValidator:
    def validate(
        self,
        request: Request,
        openapi: OpenAPI,
        body_override: bytes | None = None,
    ) -> RequestValidationResult:
        wrapped = StarletteOpenAPIRequest(request, body=body_override)
        result = openapi.unmarshal_request(wrapped)

        if result.errors:
            details = [f"{type(err).__name__}: {err}" for err in result.errors]
            raise RequestValidationError(details)

        return RequestValidationResult(
            openapi_request=wrapped,
            body=result.body,
            parameters=result.parameters,
            security=result.security,
        )

