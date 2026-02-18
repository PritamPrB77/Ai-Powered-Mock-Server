from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI

from app.openapi_loader import OperationMeta


RESERVED_PATHS = {
    "/",
    "/upload-spec",
    "/registered-endpoints",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}


@dataclass(slots=True)
class RegistrationResult:
    registered: int
    skipped: int


class DynamicRouteManager:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self._dynamic_route_names: set[str] = set()
        self.registry: dict[tuple[str, str], OperationMeta] = {}

    def clear_dynamic_routes(self) -> None:
        if self._dynamic_route_names:
            self.app.router.routes = [
                route
                for route in self.app.router.routes
                if getattr(route, "name", None) not in self._dynamic_route_names
            ]
            self._dynamic_route_names.clear()
        self.registry.clear()
        self.app.openapi_schema = None

    def register_operations(
        self,
        operations: list[OperationMeta],
        handler_factory: Callable[[OperationMeta], Callable],
    ) -> RegistrationResult:
        self.clear_dynamic_routes()
        registered = 0
        skipped = 0

        for index, operation in enumerate(operations):
            if self._is_reserved(operation.path):
                skipped += 1
                continue

            route_name = f"dynamic__{operation.method}__{index}__{operation.operation_id}"
            handler = handler_factory(operation)
            self.app.add_api_route(
                path=operation.path,
                endpoint=handler,
                methods=[operation.method.upper()],
                name=route_name,
            )
            self._dynamic_route_names.add(route_name)
            self.registry[(operation.path, operation.method)] = operation
            registered += 1

        self.app.openapi_schema = None
        return RegistrationResult(registered=registered, skipped=skipped)

    def list_registered_endpoints(self) -> list[dict[str, str]]:
        endpoints = [
            {
                "path": operation.path,
                "method": operation.method.upper(),
                "operation_id": operation.operation_id,
            }
            for operation in self.registry.values()
        ]
        return sorted(endpoints, key=lambda item: (item["path"], item["method"]))

    @staticmethod
    def _is_reserved(path: str) -> bool:
        return path in RESERVED_PATHS

