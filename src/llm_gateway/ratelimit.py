"""Per-API-key token bucket rate limiting, stored in Redis.

The whole read-refill-take-write cycle runs as one Lua script, so concurrent
gateway replicas cannot race each other into over-admitting requests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis

from llm_gateway.config import RateLimitConfig

# KEYS[1] bucket key
# ARGV: capacity, refill_per_s, now (seconds, float), cost
# Returns: {allowed (0/1), tokens_left (string), retry_after_s (string)}
TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end

local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / rate
end

redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
-- A full bucket carries no information, so let idle keys expire.
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / rate) + 60)
return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: float
    retry_after_s: float
    limit: float


class TokenBucketLimiter:
    """``now`` comes from the gateway process (injectable for tests). With several
    replicas, keep clocks NTP-synced; the alternative is ``redis.call('TIME')``
    inside the script."""

    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str = "ratelimit",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._clock = clock
        self._script = redis.register_script(TOKEN_BUCKET_LUA)

    async def acquire(
        self, key_name: str, limit: RateLimitConfig, cost: float = 1.0
    ) -> RateLimitDecision:
        allowed, tokens, retry_after = await self._script(
            keys=[f"{self._prefix}:{key_name}"],
            args=[limit.capacity, limit.refill_per_s, self._clock(), cost],
        )
        return RateLimitDecision(
            allowed=bool(int(allowed)),
            remaining=float(tokens),
            retry_after_s=float(retry_after),
            limit=limit.capacity,
        )
