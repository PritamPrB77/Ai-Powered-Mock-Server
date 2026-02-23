from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openapi_core import OpenAPI

from app.config import Settings, get_settings
from app.context_engine import ContextEngine
from app.openapi_loader import OpenAPILoader, OperationMeta
from app.request_validator import OpenAPIRequestValidator, RequestValidationError
from app.response_generator import DynamicResponseGenerator, ResponseGenerationError
from app.route_generator import DynamicRouteManager
from app.seq2seq_generator import Seq2SeqGenerator
from app.semantic_validator import SemanticValidator
from app.utils import clamp_multi_response_count


logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class RuntimeState:
    def __init__(self, app: FastAPI, settings: Settings) -> None:
        self.settings = settings
        self.loader = OpenAPILoader()
        self.route_manager = DynamicRouteManager(app)
        self.request_validator = OpenAPIRequestValidator()
        self.context_engine = ContextEngine(history_limit=settings.context_history_limit)
        self.semantic_validator = SemanticValidator(
            model_name=settings.semantic_model_name,
            threshold=settings.semantic_similarity_threshold,
        )
        self.seq2seq_generator = Seq2SeqGenerator(settings=settings)
        self.response_generator = DynamicResponseGenerator(
            settings=settings,
            semantic_validator=self.semantic_validator,
            seq2seq_generator=self.seq2seq_generator,
        )

        self.spec_dict: dict[str, Any] | None = None
        self.openapi = None

    def build_handler(self, operation: OperationMeta):
        async def dynamic_handler(request: Request) -> JSONResponse:
            return await self.handle_dynamic_request(request, operation)

        dynamic_handler.__name__ = f"dynamic_{operation.operation_id}"
        return dynamic_handler

    async def handle_dynamic_request(self, request: Request, operation: OperationMeta) -> JSONResponse:
        if self.openapi is None:
            raise HTTPException(status_code=400, detail="No OpenAPI spec loaded. Upload one first.")

        raw_body = await request.body()
        parsed_body = _parse_json_body(raw_body)
        body_n = None
        sanitized_body = parsed_body

        if isinstance(parsed_body, dict) and "n" in parsed_body:
            body_n = parsed_body.get("n")
            sanitized_body = {key: value for key, value in parsed_body.items() if key != "n"}

        n = clamp_multi_response_count(
            query_value=request.query_params.get("n"),
            body_value=body_n,
            max_n=self.settings.max_multi_response,
        )
        temperature = _parse_optional_float(request.query_params.get("temperature"))

        body_override = raw_body
        if parsed_body is not None:
            body_override = json.dumps(sanitized_body, ensure_ascii=True).encode("utf-8")

        try:
            validation = self.request_validator.validate(
                request=request,
                openapi=self.openapi,
                body_override=body_override,
            )
        except RequestValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.details) from exc

        request_body = validation.body
        parameters = validation.parameters
        path_parameters = dict(getattr(parameters, "path", {}) or {})
        query_parameters = dict(getattr(parameters, "query", {}) or {})

        if operation.method == "get":
            cached = self.context_engine.get_cached_get_response(operation.path, path_parameters)
            if cached is not None:
                payload = _shape_cached_payload(cached, n)
                self.context_engine.append_history(
                    method=operation.method,
                    path_template=operation.path,
                    request_body=request_body,
                    response_payload=payload,
                )
                return JSONResponse(status_code=operation.response_status_code, content=payload)

        context_payload = self.context_engine.get_context_for_prompt(
            path_template=operation.path,
            method=operation.method,
            path_parameters=path_parameters,
        )
        context_payload["query_parameters"] = query_parameters
        spec_info = self.spec_dict.get("info") if isinstance(self.spec_dict, dict) else {}
        if not isinstance(spec_info, dict):
            spec_info = {}
        context_payload["api_info"] = {
            "title": spec_info.get("title"),
            "description": spec_info.get("description"),
            "version": spec_info.get("version"),
        }
        context_payload["operation_summary"] = operation.summary
        context_payload["operation_description"] = operation.description
        context_payload["request_nonce"] = time.time_ns()

        try:
            generated = await self.response_generator.generate(
                operation=operation,
                openapi=self.openapi,
                openapi_request=validation.openapi_request,
                validated_request_body=request_body,
                context_payload=context_payload,
                n=n,
                temperature=temperature,
            )
        except ResponseGenerationError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        for item in generated:
            self.context_engine.update_from_response(
                method=operation.method,
                path_template=operation.path,
                path_parameters=path_parameters,
                response_payload=item,
            )

        payload: Any = generated[0] if n == 1 else generated
        self.context_engine.append_history(
            method=operation.method,
            path_template=operation.path,
            request_body=request_body,
            response_payload=payload,
        )
        return JSONResponse(status_code=operation.response_status_code, content=payload)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Dynamic Mock Server",
        description="Upload any OpenAPI spec and auto-generate context-aware dynamic mock endpoints.",
        version="1.0.0",
    )
    settings = get_settings()
    runtime = RuntimeState(app, settings)
    app.state.runtime = runtime

    @app.on_event("startup")
    async def startup_warmup() -> None:
        if runtime.settings.seq2seq_enabled and runtime.settings.seq2seq_warmup_on_startup:
            loaded = await runtime.seq2seq_generator.warmup()
            if loaded:
                logger.info("Seq2Seq model warmed up.")
            else:
                logger.warning("Seq2Seq warmup skipped or failed; runtime will use available fallbacks.")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/registered-endpoints")
    async def registered_endpoints() -> dict[str, Any]:
        endpoints = runtime.route_manager.list_registered_endpoints()
        return {"count": len(endpoints), "endpoints": endpoints}

    @app.post("/upload-spec")
    async def upload_spec(
        request: Request,
        spec_file: UploadFile | None = File(default=None),
        spec_text: str | None = Form(default=None),
    ) -> dict[str, Any]:
        payload = await _extract_spec_payload(request, spec_file, spec_text)
        try:
            spec_dict, openapi, operations = runtime.loader.load(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid OpenAPI spec: {exc}") from exc

        spec_dict = _ensure_runtime_server(spec_dict, request)
        openapi = OpenAPI.from_dict(spec_dict)

        runtime.spec_dict = spec_dict
        runtime.openapi = openapi
        runtime.context_engine.clear()
        registration = runtime.route_manager.register_operations(
            operations=operations,
            handler_factory=runtime.build_handler,
        )

        endpoints = runtime.route_manager.list_registered_endpoints()
        return {
            "message": "OpenAPI spec uploaded successfully.",
            "registered": registration.registered,
            "skipped": registration.skipped,
            "endpoints": endpoints,
        }

    return app


app = create_app()


def _parse_json_body(raw_body: bytes) -> Any | None:
    if not raw_body:
        return None
    text = raw_body.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def _extract_spec_payload(
    request: Request,
    spec_file: UploadFile | None,
    spec_text: str | None,
) -> str | bytes | dict[str, Any]:
    if spec_file is not None:
        file_bytes = await spec_file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded spec file is empty.")
        return file_bytes

    if spec_text is not None and spec_text.strip():
        return spec_text

    raw_body = await request.body()
    if not raw_body:
        raise HTTPException(
            status_code=400,
            detail="No OpenAPI payload found. Provide `spec_file`, `spec_text`, or raw body.",
        )

    parsed_body = _parse_json_body(raw_body)
    if isinstance(parsed_body, dict) and "spec" in parsed_body:
        return parsed_body["spec"]
    if isinstance(parsed_body, dict) and "openapi" in parsed_body and "paths" in parsed_body:
        return parsed_body

    return raw_body


def _shape_cached_payload(cached_payload: Any, n: int) -> Any:
    if n == 1:
        return deepcopy(cached_payload)
    if isinstance(cached_payload, list):
        if not cached_payload:
            return []
        if len(cached_payload) >= n:
            return deepcopy(cached_payload[:n])
        padded = deepcopy(cached_payload)
        while len(padded) < n:
            padded.append(deepcopy(padded[-1]))
        return padded
    return [deepcopy(cached_payload) for _ in range(n)]


def _ensure_runtime_server(spec_dict: dict[str, Any], request: Request) -> dict[str, Any]:
    runtime_url = str(request.base_url).rstrip("/")
    servers = spec_dict.get("servers")
    if not isinstance(servers, list):
        servers = []

    known_urls: set[str] = set()
    for server in servers:
        if not isinstance(server, dict):
            continue
        url = server.get("url")
        if isinstance(url, str):
            known_urls.add(url.rstrip("/"))

    if runtime_url not in known_urls:
        servers.append({"url": runtime_url})

    spec_dict["servers"] = servers
    return spec_dict
