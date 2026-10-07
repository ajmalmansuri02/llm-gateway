"""The request pipeline: cache -> route (retry/fallback) -> account -> cache store.

Kept separate from the HTTP layer so it can be tested and reasoned about without
FastAPI in the way.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llm_gateway import metrics
from llm_gateway.accounting import Usage, UsageTracker
from llm_gateway.cache import ExactCache, SemanticCache, exact_cache_key
from llm_gateway.config import GatewayConfig
from llm_gateway.providers import Provider, ProviderError
from llm_gateway.providers.base import estimate_tokens, new_completion_id, now_ts
from llm_gateway.redaction import redact, redact_obj
from llm_gateway.routing import Attempted, Route, Router, aclose_quietly
from llm_gateway.schemas import ChatCompletionRequest, EmbeddingRequest
from llm_gateway.tracing import tracer

log = logging.getLogger("llm_gateway.requests")


@dataclass
class ResultMeta:
    alias: str
    provider: str = "cache"
    model: str = ""
    cache: str = "miss"  # "miss" | "hit" | "semantic-hit" | "bypass"
    fallbacks: int = 0
    usage: Usage | None = None
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def headers(self) -> dict[str, str]:
        return {
            "x-gateway-provider": self.provider,
            "x-gateway-upstream-model": self.model,
            "x-gateway-cache": self.cache,
            "x-gateway-fallbacks": str(self.fallbacks),
        }


class ChatService:
    def __init__(
        self,
        config: GatewayConfig,
        router: Router,
        usage: UsageTracker,
        exact_cache: ExactCache | None,
        semantic_cache: SemanticCache | None,
    ) -> None:
        self.config = config
        self.router = router
        self.usage = usage
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache

    # ---------------------------------------------------------------- helpers
    def _log_exchange(
        self, key_name: str, request: ChatCompletionRequest, text: str, meta: ResultMeta
    ) -> None:
        extra: dict[str, Any] = {
            "api_key": key_name,
            "model_alias": meta.alias,
            "provider": meta.provider,
            "upstream_model": meta.model,
            "cache": meta.cache,
            "fallbacks": meta.fallbacks,
            "prompt_tokens": meta.usage.prompt_tokens if meta.usage else 0,
            "completion_tokens": meta.usage.completion_tokens if meta.usage else 0,
            "cost_usd": meta.cost_usd,
        }
        if self.config.logging.log_bodies:
            messages = [m.model_dump(exclude_none=True) for m in request.messages]
            if self.config.logging.redact_pii:
                messages, text = redact_obj(messages), redact(text)
            extra["messages"] = messages
            extra["completion"] = text
        log.info("chat completion", extra=extra)

    def _cache_enabled(self, route: Route, bypass: bool) -> bool:
        return self.config.cache.enabled and route.model.cache and not bypass

    def _semantic_enabled(self, route: Route, bypass: bool) -> bool:
        return (
            self._cache_enabled(route, bypass)
            and self.semantic_cache is not None
            and route.model.semantic_cache
        )

    async def _cache_lookup(
        self, route: Route, request: ChatCompletionRequest, bypass: bool
    ) -> tuple[dict[str, Any] | None, str, str | None, list[float] | None]:
        """Returns (response, cache status, exact key, prompt vector)."""
        if not self._cache_enabled(route, bypass):
            return None, "bypass", None, None
        key = exact_cache_key(route.alias, request)
        assert self.exact_cache is not None
        try:
            cached = await self.exact_cache.get(key)
        except Exception:
            log.warning("exact cache lookup failed", exc_info=True)
            cached = None
        metrics.CACHE_LOOKUPS.labels("exact", "hit" if cached else "miss").inc()
        if cached:
            return cached, "hit", key, None

        vector = None
        if self._semantic_enabled(route, bypass):
            assert self.semantic_cache is not None
            try:
                cached, vector = await self.semantic_cache.lookup(route.alias, request)
            except Exception:
                # The semantic cache is an optimisation; never fail a request over it.
                log.warning("semantic cache lookup failed", exc_info=True)
                cached = None
            metrics.CACHE_LOOKUPS.labels("semantic", "hit" if cached else "miss").inc()
            if cached:
                return cached, "semantic-hit", key, vector
        return None, "miss", key, vector

    async def _cache_store(
        self,
        route: Route,
        key: str | None,
        vector: list[float] | None,
        response: dict[str, Any],
    ) -> None:
        if key is None or self.exact_cache is None:
            return
        try:
            await self.exact_cache.set(key, response)
            if vector is not None and self.semantic_cache is not None:
                await self.semantic_cache.store(route.alias, vector, response)
        except Exception:
            log.warning("cache store failed", exc_info=True)

    # ------------------------------------------------------- non-streaming
    async def complete(
        self, request: ChatCompletionRequest, key_name: str, *, bypass_cache: bool = False
    ) -> tuple[dict[str, Any], ResultMeta]:
        route = self.router.resolve(request.model, "chat")
        meta = ResultMeta(alias=route.alias)
        with tracer.start_as_current_span(
            "gateway.chat", attributes={"gateway.model_alias": route.alias, "gateway.stream": False}
        ) as span:
            cached, meta.cache, key, vector = await self._cache_lookup(route, request, bypass_cache)
            if cached is not None:
                response = copy.deepcopy(cached)
                response["id"] = new_completion_id()
                response["created"] = now_ts()
                meta.usage = Usage.from_openai(response.get("usage"))
                await self.usage.record(key_name, "cache", route.alias, Usage(0, 0), cached=True)
            else:
                response, attempted = await self.router.run(
                    route, lambda provider, model: provider.chat(model, request)
                )
                self._apply_attempt(meta, attempted)
                text = _completion_text(response)
                meta.usage = Usage.from_openai(response.get("usage")) or Usage(
                    estimate_tokens(request.prompt_text()), estimate_tokens(text)
                )
                response["usage"] = meta.usage.as_openai()
                meta.cost_usd = await self.usage.record(
                    key_name, meta.provider, meta.model, meta.usage
                )
                await self._cache_store(route, key, vector, response)
            response["model"] = route.alias
            span.set_attribute("gateway.cache", meta.cache)
            span.set_attribute("gen_ai.system", meta.provider)
            if meta.usage:
                span.set_attribute("gen_ai.usage.input_tokens", meta.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", meta.usage.completion_tokens)
        self._log_exchange(key_name, request, _completion_text(response), meta)
        return response, meta

    @staticmethod
    def _apply_attempt(meta: ResultMeta, attempted: Attempted) -> None:
        meta.provider = attempted.provider
        meta.model = attempted.model
        meta.fallbacks = attempted.fallbacks

    # ----------------------------------------------------------- streaming
    async def stream(
        self, request: ChatCompletionRequest, key_name: str, *, bypass_cache: bool = False
    ) -> tuple[AsyncIterator[str], ResultMeta]:
        """Returns an SSE string iterator plus metadata known before the first byte
        (provider and cache status) so they can go into response headers."""
        route = self.router.resolve(request.model, "chat")
        meta = ResultMeta(alias=route.alias)
        include_usage = bool(request.stream_options and request.stream_options.include_usage)
        started = time.perf_counter()

        cached, meta.cache, key, vector = await self._cache_lookup(route, request, bypass_cache)
        if cached is not None:
            await self.usage.record(key_name, "cache", route.alias, Usage(0, 0), cached=True)
            meta.usage = Usage.from_openai(cached.get("usage"))
            return _replay_as_stream(cached, route.alias, include_usage), meta

        first, rest, attempted = await self.router.open_stream(route, request)
        self._apply_attempt(meta, attempted)
        metrics.TIME_TO_FIRST_TOKEN.labels(route.alias, meta.provider).observe(
            time.perf_counter() - started
        )

        async def events() -> AsyncIterator[str]:
            try:
                async for event in _events():
                    yield event
            finally:
                # Also runs when the client disconnects mid-stream.
                await aclose_quietly(rest)

        async def _events() -> AsyncIterator[str]:
            parts: list[str] = []
            upstream_usage: dict[str, Any] | None = None
            finish_reason = "stop"
            completion_id = first.get("id") or new_completion_id()
            created = first.get("created") or now_ts()
            with tracer.start_as_current_span(
                "gateway.chat.stream",
                attributes={"gateway.model_alias": route.alias, "gen_ai.system": meta.provider},
            ) as span:
                chunk: dict[str, Any] | None = first
                try:
                    while chunk is not None:
                        if chunk.get("usage"):
                            upstream_usage = chunk["usage"]
                        for choice in chunk.get("choices") or []:
                            parts.append((choice.get("delta") or {}).get("content") or "")
                            finish_reason = choice.get("finish_reason") or finish_reason
                        if chunk.get("choices"):
                            out = {k: v for k, v in chunk.items() if k != "usage"}
                            out["model"] = route.alias
                            yield _sse(out)
                        try:
                            chunk = await anext(rest)
                        except StopAsyncIteration:
                            chunk = None
                except ProviderError as exc:
                    await aclose_quietly(rest)
                    # Too late to fall back: the client already has partial output.
                    span.record_exception(exc)
                    metrics.PROVIDER_ERRORS.labels(
                        meta.provider, exc.kind, str(exc.status_code or "")
                    ).inc()
                    yield _sse(
                        {"error": {"message": str(exc), "type": "upstream_error", "code": None}}
                    )
                    yield "data: [DONE]\n\n"
                    return

                text = "".join(parts)
                meta.usage = Usage.from_openai(upstream_usage) or Usage(
                    estimate_tokens(request.prompt_text()), estimate_tokens(text)
                )
                meta.cost_usd = await self.usage.record(
                    key_name, meta.provider, meta.model, meta.usage
                )
                if include_usage:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": route.alias,
                            "choices": [],
                            "usage": meta.usage.as_openai(),
                        }
                    )
                yield "data: [DONE]\n\n"
                await self._cache_store(
                    route,
                    key,
                    vector,
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": route.alias,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": meta.usage.as_openai(),
                    },
                )
                self._log_exchange(key_name, request, text, meta)

        return events(), meta

    # ---------------------------------------------------------- embeddings
    async def embed(
        self, request: EmbeddingRequest, key_name: str
    ) -> tuple[dict[str, Any], ResultMeta]:
        route = self.router.resolve(request.model, "embedding")
        meta = ResultMeta(alias=route.alias)
        inputs = request.inputs()
        with tracer.start_as_current_span(
            "gateway.embeddings", attributes={"gateway.model_alias": route.alias}
        ):
            result, attempted = await self.router.run(
                route, lambda provider, model: provider.embed(model, inputs)
            )
        self._apply_attempt(meta, attempted)
        meta.cache = "bypass"
        meta.usage = Usage(result.prompt_tokens, 0)
        meta.cost_usd = await self.usage.record(key_name, meta.provider, meta.model, meta.usage)
        return (
            {
                "object": "list",
                "model": route.alias,
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector}
                    for i, vector in enumerate(result.vectors)
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "total_tokens": result.prompt_tokens,
                },
            },
            meta,
        )


def make_embedder(router: Router, alias: str):  # type: ignore[no-untyped-def]
    """An embedding function for the semantic cache that reuses routing/fallback."""

    async def embed(text: str) -> list[float]:
        route = router.resolve(alias, "embedding")

        async def call(provider: Provider, model: str) -> list[float]:
            return (await provider.embed(model, [text])).vectors[0]

        vector, _ = await router.run(route, call)
        return vector

    return embed


def _completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    return content if isinstance(content, str) else ""


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def _replay_as_stream(
    cached: dict[str, Any], alias: str, include_usage: bool
) -> AsyncIterator[str]:
    completion_id, created = new_completion_id(), now_ts()
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": alias,
    }
    text = _completion_text(cached)
    finish = (cached.get("choices") or [{}])[0].get("finish_reason") or "stop"
    yield _sse(
        {
            **base,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
            ],
        }
    )
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if include_usage and cached.get("usage"):
        yield _sse({**base, "choices": [], "usage": cached["usage"]})
    yield "data: [DONE]\n\n"
