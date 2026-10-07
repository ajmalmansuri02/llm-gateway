"""FastAPI application: wiring and the HTTP surface."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry import propagate
from opentelemetry.trace import SpanKind
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from llm_gateway import __version__, metrics
from llm_gateway.accounting import UsageTracker
from llm_gateway.auth import Authenticator, Principal
from llm_gateway.cache import ExactCache, SemanticCache
from llm_gateway.config import GatewayConfig, Settings, load_gateway_config
from llm_gateway.errors import GatewayError, RateLimitedError
from llm_gateway.logging_setup import setup_logging
from llm_gateway.providers import Provider, build_provider
from llm_gateway.ratelimit import TokenBucketLimiter
from llm_gateway.routing import Router
from llm_gateway.schemas import ChatCompletionRequest, EmbeddingRequest
from llm_gateway.service import ChatService, make_embedder
from llm_gateway.tracing import setup_tracing, tracer

log = logging.getLogger("llm_gateway")


@dataclass
class GatewayState:
    settings: Settings
    config: GatewayConfig
    redis: Redis
    providers: dict[str, Provider]
    router: Router
    service: ChatService
    limiter: TokenBucketLimiter
    auth: Authenticator


def _redis_from_url(url: str) -> Redis:
    if url.startswith("memory://"):
        # Handy for trying the gateway without Redis; state is per-process only.
        import fakeredis

        return fakeredis.FakeAsyncRedis()
    return Redis.from_url(url)


def build_state(
    settings: Settings,
    config: GatewayConfig,
    *,
    redis: Redis | None = None,
    providers: dict[str, Provider] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
) -> GatewayState:
    redis = redis if redis is not None else _redis_from_url(settings.redis_url)
    if providers is None:
        providers = {name: build_provider(name, cfg) for name, cfg in config.providers.items()}
    router = Router(config, providers, **({"sleep": sleep} if sleep else {}))
    semantic = None
    if config.cache.enabled and config.cache.semantic.enabled:
        sem = config.cache.semantic
        semantic = SemanticCache(
            redis,
            make_embedder(router, sem.embedding_model),
            threshold=sem.threshold,
            max_entries=sem.max_entries,
            ttl_s=config.cache.ttl_s,
        )
    service = ChatService(
        config,
        router,
        UsageTracker(redis, config),
        ExactCache(redis, config.cache.ttl_s) if config.cache.enabled else None,
        semantic,
    )
    limiter = TokenBucketLimiter(redis, **({"clock": clock} if clock else {}))
    return GatewayState(
        settings, config, redis, providers, router, service, limiter, Authenticator(config.auth)
    )


def create_app(
    settings: Settings | None = None,
    config: GatewayConfig | None = None,
    *,
    state: GatewayState | None = None,
) -> FastAPI:
    settings = settings or (state.settings if state else Settings())
    if state is None:
        setup_logging(settings.log_level, settings.log_json)
        setup_tracing(settings.otel_exporter_otlp_endpoint, settings.otel_service_name)
        config = config or load_gateway_config(settings.gateway_config_path)
        state = build_state(settings, config)
    gw = state

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info(
            "gateway started",
            extra={"models": sorted(gw.config.models), "providers": sorted(gw.providers)},
        )
        yield
        for provider in gw.providers.values():
            await provider.aclose()
        await gw.redis.aclose()

    app = FastAPI(title="llm-gateway", version=__version__, lifespan=lifespan)
    app.state.gateway = gw

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        ctx = propagate.extract(dict(request.headers))
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            context=ctx,
            kind=SpanKind.SERVER,
            attributes={"http.request.method": request.method, "url.path": request.url.path},
        ) as span:
            response = await call_next(request)
            span.set_attribute("http.response.status_code", response.status_code)
        response.headers["x-request-id"] = request_id
        if request.url.path not in ("/metrics", "/healthz", "/readyz"):
            log.info(
                "http request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "api_key": getattr(request.state, "api_key_name", None),
                },
            )
        return response

    # -------------------------------------------------------- error handling
    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(exc.to_body(), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return JSONResponse(
            {
                "error": {
                    "message": f"{loc}: {first.get('msg', 'invalid request')}",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
            status_code=400,
        )

    # ---------------------------------------------------------- dependencies
    async def principal(request: Request) -> Principal:
        who = gw.auth.authenticate(request.headers.get("authorization"))
        request.state.api_key_name = who.name
        decision = await gw.limiter.acquire(who.name, who.rate_limit)
        request.state.rate_limit = decision
        if not decision.allowed:
            metrics.RATE_LIMITED.labels(who.name).inc()
            raise RateLimitedError(
                f"Rate limit exceeded for key '{who.name}'. Retry in "
                f"{decision.retry_after_s:.2f}s.",
                code="rate_limit_exceeded",
                headers={
                    "retry-after": str(max(1, round(decision.retry_after_s + 0.5))),
                    "x-ratelimit-limit-requests": str(int(decision.limit)),
                    "x-ratelimit-remaining-requests": "0",
                },
            )
        return who

    def _bypass_cache(request: Request) -> bool:
        cache_control = request.headers.get("cache-control", "").lower()
        return "no-cache" in cache_control or "no-store" in cache_control

    def _headers(request: Request, extra: dict[str, str]) -> dict[str, str]:
        decision = request.state.rate_limit
        return {
            **extra,
            "x-ratelimit-limit-requests": str(int(decision.limit)),
            "x-ratelimit-remaining-requests": str(int(decision.remaining)),
        }

    # --------------------------------------------------------------- routes
    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest, request: Request, who: Principal = Depends(principal)
    ) -> Response:
        started = time.perf_counter()
        status = "200"
        provider = "none"
        # Only configured aliases become label values (bounded cardinality).
        alias = body.model if body.model in gw.config.models else "other"
        try:
            if body.stream:
                events, meta = await gw.service.stream(
                    body, who.name, bypass_cache=_bypass_cache(request)
                )
                provider = meta.provider

                async def timed() -> AsyncIterator[str]:
                    try:
                        async for event in events:
                            yield event
                    finally:
                        metrics.REQUEST_LATENCY.labels(
                            "chat_stream", alias, meta.provider, "200"
                        ).observe(time.perf_counter() - started)

                return StreamingResponse(
                    timed(),
                    media_type="text/event-stream",
                    headers=_headers(
                        request,
                        {**meta.headers(), "cache-control": "no-cache", "x-accel-buffering": "no"},
                    ),
                )
            response, meta = await gw.service.complete(
                body, who.name, bypass_cache=_bypass_cache(request)
            )
            provider = meta.provider
            return JSONResponse(response, headers=_headers(request, meta.headers()))
        except GatewayError as exc:
            status = str(exc.status_code)
            raise
        finally:
            if not body.stream or status != "200":
                metrics.REQUEST_LATENCY.labels("chat", alias, provider, status).observe(
                    time.perf_counter() - started
                )

    @app.post("/v1/embeddings")
    async def embeddings(
        body: EmbeddingRequest, request: Request, who: Principal = Depends(principal)
    ) -> JSONResponse:
        started = time.perf_counter()
        response, meta = await gw.service.embed(body, who.name)
        metrics.REQUEST_LATENCY.labels("embeddings", meta.alias, meta.provider, "200").observe(
            time.perf_counter() - started
        )
        return JSONResponse(response, headers=_headers(request, meta.headers()))

    @app.get("/v1/models")
    async def list_models(_: Principal = Depends(principal)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": alias, "object": "model", "owned_by": "llm-gateway", "kind": m.kind}
                for alias, m in sorted(gw.config.models.items())
            ],
        }

    @app.get("/v1/usage")
    async def usage(who: Principal = Depends(principal)) -> dict[str, Any]:
        """Token and spend totals for the calling API key."""
        return await gw.service.usage.get(who.name)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up and serving."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness: dependencies needed to serve traffic are reachable."""
        checks: dict[str, str] = {}
        try:
            await gw.redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {type(exc).__name__}"
        ready = all(v == "ok" for v in checks.values())
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "checks": checks},
            status_code=200 if ready else 503,
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def app_from_env() -> FastAPI:
    """Entry point for ``uvicorn llm_gateway.app:app_from_env --factory``."""
    return create_app()
