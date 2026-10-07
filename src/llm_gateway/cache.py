"""Response caches.

* ``ExactCache`` - key is a hash of everything that affects the output. Cheap and
  always correct, but only hits on byte-identical requests.
* ``SemanticCache`` - embeds the prompt and returns a stored response when a
  previous prompt is "close enough" (cosine similarity >= threshold). Higher hit
  rate, but it can return an answer to a slightly different question, so it is
  opt-in per model alias.

The semantic index is a capped Redis list scanned linearly. That is fine for a
few thousand entries; past that you want a real vector index (Redis Stack's
``FT.SEARCH`` with HNSW, pgvector, ...).
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from redis.asyncio import Redis

from llm_gateway.schemas import ChatCompletionRequest

# Fields that never change the model's output.
_IGNORED_FIELDS = {"stream", "stream_options", "user", "model"}


def exact_cache_key(alias: str, request: ChatCompletionRequest) -> str:
    body = request.model_dump(exclude_none=True, exclude=_IGNORED_FIELDS)
    canonical = json.dumps({"alias": alias, "body": body}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class ExactCache:
    def __init__(self, redis: Redis, ttl_s: int, prefix: str = "cache:exact") -> None:
        self._redis = redis
        self._ttl = ttl_s
        self._prefix = prefix

    async def get(self, key: str) -> dict[str, Any] | None:
        raw = await self._redis.get(f"{self._prefix}:{key}")
        return json.loads(raw) if raw else None

    async def set(self, key: str, response: dict[str, Any]) -> None:
        await self._redis.set(f"{self._prefix}:{key}", json.dumps(response), ex=self._ttl)


Embedder = Callable[[str], Awaitable[list[float]]]


class SemanticCache:
    def __init__(
        self,
        redis: Redis,
        embedder: Embedder,
        *,
        threshold: float,
        max_entries: int,
        ttl_s: int,
        prefix: str = "cache:sem",
    ) -> None:
        self._redis = redis
        self._embed = embedder
        self.threshold = threshold
        self._max = max_entries
        self._ttl = ttl_s
        self._prefix = prefix

    @staticmethod
    def prompt_of(request: ChatCompletionRequest) -> str:
        return request.prompt_text()

    async def lookup(
        self, alias: str, request: ChatCompletionRequest
    ) -> tuple[dict[str, Any] | None, list[float]]:
        """Returns (cached response or None, the prompt embedding for reuse on store)."""
        vector = await self._embed(self.prompt_of(request))
        entries = await self._redis.lrange(f"{self._prefix}:index:{alias}", 0, self._max - 1)
        best_id, best_score = None, -1.0
        for raw in entries:
            entry = json.loads(raw)
            score = cosine(vector, entry["v"])
            if score > best_score:
                best_id, best_score = entry["id"], score
        if best_id is not None and best_score >= self.threshold:
            raw_response = await self._redis.get(f"{self._prefix}:resp:{best_id}")
            if raw_response:
                return json.loads(raw_response), vector
        return None, vector

    async def store(self, alias: str, vector: list[float], response: dict[str, Any]) -> None:
        entry_id = uuid.uuid4().hex
        index_key = f"{self._prefix}:index:{alias}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(f"{self._prefix}:resp:{entry_id}", json.dumps(response), ex=self._ttl)
            pipe.lpush(index_key, json.dumps({"id": entry_id, "v": vector}))
            pipe.ltrim(index_key, 0, self._max - 1)
            pipe.expire(index_key, self._ttl)
            await pipe.execute()
