"""Model routing, retries and fallback.

``resolve`` turns a client-facing alias into an ordered list of targets.
``run`` walks that list: each target gets up to ``retry.max_attempts`` tries with
exponential backoff + jitter for retryable errors, then the next target is tried.
Errors caused by the request itself (HTTP 400/413/422) stop the walk at once,
because no other provider will accept a malformed request either.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from opentelemetry.trace import Status, StatusCode

from llm_gateway import metrics
from llm_gateway.config import GatewayConfig, ModelConfig, TargetConfig
from llm_gateway.errors import InvalidRequestError, ModelNotFoundError, UpstreamError
from llm_gateway.providers import Provider, ProviderError
from llm_gateway.tracing import tracer

log = logging.getLogger(__name__)

T = TypeVar("T")
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class Route:
    alias: str
    model: ModelConfig


@dataclass(frozen=True)
class Attempted:
    """Which target finally served the request."""

    provider: str
    model: str
    fallbacks: int


class Router:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        *,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.providers = providers
        self._sleep = sleep
        self._rng = rng or random.Random()

    def resolve(self, alias: str, kind: str = "chat") -> Route:
        model = self.config.models.get(alias)
        if model is None and self.config.allow_direct_models and "/" in alias:
            provider, _, upstream_model = alias.partition("/")
            if provider in self.providers:
                model = ModelConfig(
                    kind=kind,  # type: ignore[arg-type]
                    targets=[TargetConfig(provider=provider, model=upstream_model)],
                )
        if model is None:
            raise ModelNotFoundError(f"The model '{alias}' does not exist", code="model_not_found")
        if model.kind != kind:
            raise InvalidRequestError(
                f"The model '{alias}' is a {model.kind} model, not a {kind} model",
                code="wrong_model_kind",
            )
        return Route(alias, model)

    def _timeout_for(self, target: TargetConfig) -> float:
        return target.timeout_s or self.providers[target.provider].timeout_s

    def _backoff(self, attempt: int) -> float:
        retry = self.config.retry
        delay = min(retry.max_delay_s, retry.base_delay_s * (2**attempt))
        return delay * self._rng.uniform(0.5, 1.0)  # "equal jitter"

    async def _attempt(
        self,
        target: TargetConfig,
        call: Callable[[Provider, str], Awaitable[T]],
    ) -> T:
        provider = self.providers[target.provider]
        attempts = self.config.retry.max_attempts
        for attempt in range(attempts):
            with tracer.start_as_current_span(
                "provider.call",
                attributes={
                    "gen_ai.system": target.provider,
                    "gen_ai.request.model": target.model,
                    "gateway.attempt": attempt + 1,
                },
            ) as span:
                try:
                    async with asyncio.timeout(self._timeout_for(target)):
                        result = await call(provider, target.model)
                except TimeoutError:
                    error = ProviderError(
                        f"no response within {self._timeout_for(target)}s",
                        provider=target.provider,
                        kind="timeout",
                    )
                except ProviderError as exc:
                    error = exc
                else:
                    metrics.PROVIDER_REQUESTS.labels(target.provider, target.model, "success").inc()
                    return result

                span.set_status(Status(StatusCode.ERROR, str(error)))
                metrics.PROVIDER_REQUESTS.labels(target.provider, target.model, "error").inc()
                metrics.PROVIDER_ERRORS.labels(
                    target.provider, error.kind, str(error.status_code or "")
                ).inc()
                log.warning(
                    "provider call failed",
                    extra={
                        "provider": target.provider,
                        "model": target.model,
                        "attempt": attempt + 1,
                        "error": str(error),
                    },
                )
                if not error.retryable or attempt == attempts - 1:
                    raise error
                await self._sleep(self._backoff(attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    async def run(
        self,
        route: Route,
        call: Callable[[Provider, str], Awaitable[T]],
    ) -> tuple[T, Attempted]:
        errors: list[ProviderError] = []
        for index, target in enumerate(route.model.targets):
            try:
                result = await self._attempt(target, call)
                return result, Attempted(target.provider, target.model, index)
            except ProviderError as exc:
                errors.append(exc)
                if not exc.should_fallback:
                    raise InvalidRequestError(
                        f"Upstream rejected the request: {exc.message}",
                        status_code=exc.status_code,
                        code="upstream_rejected",
                    ) from exc
                if index < len(route.model.targets) - 1:
                    metrics.FALLBACKS.labels(route.alias, target.provider).inc()
        raise UpstreamError(
            "All providers failed: " + "; ".join(str(e) for e in errors),
            code="all_providers_failed",
        )

    async def open_stream(
        self, route: Route, request: Any
    ) -> tuple[dict[str, Any], AsyncIterator[dict[str, Any]], Attempted]:
        """Start a stream with fallback.

        Fallback is only possible *before* the first chunk reaches the client, so
        this pulls the first chunk inside the retry loop and hands back
        ``(first_chunk, rest_of_stream, attempted)``.
        """

        async def first_chunk(
            provider: Provider, model: str
        ) -> tuple[dict[str, Any], AsyncIterator[dict[str, Any]]]:
            iterator = aiter(provider.stream_chat(model, request))
            try:
                chunk = await anext(iterator)
            except StopAsyncIteration as exc:
                raise ProviderError(
                    "stream ended before any data", provider=provider.name, kind="upstream"
                ) from exc
            except BaseException:
                await aclose_quietly(iterator)
                raise
            return chunk, iterator

        (chunk, iterator), attempted = await self.run(route, first_chunk)
        return chunk, iterator, attempted


async def aclose_quietly(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception:  # pragma: no cover - best effort cleanup
            log.debug("error closing upstream stream", exc_info=True)
