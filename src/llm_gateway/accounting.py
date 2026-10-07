"""Token and cost accounting per API key.

Totals live in Redis hashes so they survive restarts and are shared by replicas;
the same numbers are also exported as Prometheus counters for dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis

from llm_gateway import metrics
from llm_gateway.config import GatewayConfig


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_openai(cls, usage: dict[str, Any] | None) -> Usage | None:
        if not usage:
            return None
        return cls(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))

    def as_openai(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class UsageTracker:
    def __init__(self, redis: Redis, config: GatewayConfig, prefix: str = "usage") -> None:
        self._redis = redis
        self._config = config
        self._prefix = prefix

    def cost_usd(self, provider: str, model: str, usage: Usage) -> float:
        price = self._config.price_for(provider, model)
        return (
            usage.prompt_tokens * price.input_per_1m + usage.completion_tokens * price.output_per_1m
        ) / 1_000_000

    async def record(
        self,
        key_name: str,
        provider: str,
        model: str,
        usage: Usage,
        *,
        cached: bool = False,
    ) -> float:
        """Record a request; returns its cost. Cached responses cost nothing upstream,
        so they count as requests and cache hits but not tokens or spend."""
        key = f"{self._prefix}:{key_name}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hincrby(key, "requests", 1)
            if cached:
                pipe.hincrby(key, "cache_hits", 1)
                await pipe.execute()
                return 0.0
            cost = self.cost_usd(provider, model, usage)
            pipe.hincrby(key, "prompt_tokens", usage.prompt_tokens)
            pipe.hincrby(key, "completion_tokens", usage.completion_tokens)
            pipe.hincrbyfloat(key, "cost_usd", cost)
            pipe.hincrby(key, f"model:{provider}/{model}:tokens", usage.total_tokens)
            await pipe.execute()

        metrics.TOKENS.labels(provider, model, key_name, "prompt").inc(usage.prompt_tokens)
        metrics.TOKENS.labels(provider, model, key_name, "completion").inc(usage.completion_tokens)
        metrics.COST_USD.labels(provider, model, key_name).inc(cost)
        return cost

    async def get(self, key_name: str) -> dict[str, Any]:
        raw = await self._redis.hgetall(f"{self._prefix}:{key_name}")
        data = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in raw.items()
        }
        by_model = {
            k.removeprefix("model:").removesuffix(":tokens"): int(v)
            for k, v in data.items()
            if k.startswith("model:")
        }
        return {
            "api_key": key_name,
            "requests": int(data.get("requests", 0)),
            "cache_hits": int(data.get("cache_hits", 0)),
            "prompt_tokens": int(data.get("prompt_tokens", 0)),
            "completion_tokens": int(data.get("completion_tokens", 0)),
            "cost_usd": round(float(data.get("cost_usd", 0.0)), 8),
            "tokens_by_model": by_model,
        }
