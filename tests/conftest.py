"""Test fixtures: a fully wired gateway backed by fakeredis and fake providers.

No network, no Redis server, no model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import fakeredis
import httpx
import pytest

from llm_gateway.app import GatewayState, build_state, create_app
from llm_gateway.config import GatewayConfig, Settings
from llm_gateway.providers import FakeProvider, Provider

API_KEY = "sk-test-key-123456"


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def base_config(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "providers": {
            "primary": {"type": "fake", "timeout_s": 1},
            "secondary": {"type": "fake", "timeout_s": 1},
            "embedder": {"type": "fake"},
        },
        "models": {
            "chat": {
                "targets": [
                    {"provider": "primary", "model": "p-model"},
                    {"provider": "secondary", "model": "s-model"},
                ]
            },
            "chat-sem": {
                "targets": [{"provider": "primary", "model": "p-model"}],
                "semantic_cache": True,
            },
            "embed": {"kind": "embedding", "targets": [{"provider": "embedder", "model": "e"}]},
        },
        "pricing": {
            "primary/p-model": {"input_per_1m": 1.0, "output_per_1m": 2.0},
            "secondary/*": {"input_per_1m": 10.0, "output_per_1m": 20.0},
        },
        "auth": {
            "enabled": True,
            "keys": [
                {"name": "tester", "key": API_KEY},
                {
                    "name": "tiny",
                    "key": "sk-tiny-key-123456",
                    "rate_limit": {"capacity": 2, "refill_per_s": 1},
                },
            ],
        },
        "retry": {"max_attempts": 2, "base_delay_s": 0, "max_delay_s": 0},
        "cache": {
            "enabled": True,
            "ttl_s": 60,
            "semantic": {"enabled": True, "embedding_model": "embed", "threshold": 0.9},
        },
    }
    for key, value in overrides.items():
        raw[key] = value
    return raw


@dataclass
class Harness:
    client: httpx.AsyncClient
    state: GatewayState
    providers: dict[str, FakeProvider]
    clock: FakeClock
    sleeps: list[float]

    async def chat(self, content: str = "hello", key: str = API_KEY, **body: Any) -> httpx.Response:
        payload = {"model": "chat", "messages": [{"role": "user", "content": content}], **body}
        return await self.client.post(
            "/v1/chat/completions", json=payload, headers={"authorization": f"Bearer {key}"}
        )


MakeHarness = Callable[..., Any]


@pytest.fixture
async def make_harness() -> AsyncIterator[MakeHarness]:
    clients: list[httpx.AsyncClient] = []

    async def factory(
        providers: dict[str, Provider] | None = None, **config_overrides: Any
    ) -> Harness:
        config = GatewayConfig.model_validate(base_config(**config_overrides))
        fakes: dict[str, Any] = {
            "primary": FakeProvider("primary", timeout_s=1),
            "secondary": FakeProvider("secondary", timeout_s=1),
            "embedder": FakeProvider("embedder"),
        }
        fakes.update(providers or {})
        clock = FakeClock()
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        state = build_state(
            Settings(redis_url="redis://unused"),
            config,
            redis=fakeredis.FakeAsyncRedis(),
            providers=fakes,
            sleep=fake_sleep,
            clock=clock,
        )
        app = create_app(state=state)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
        clients.append(client)
        return Harness(client, state, fakes, clock, sleeps)

    yield factory
    for client in clients:
        await client.aclose()


@pytest.fixture
async def harness(make_harness: MakeHarness) -> Harness:
    return await make_harness()


def parse_sse(text: str) -> list[Any]:
    import json

    events: list[Any] = []
    for block in text.strip().split("\n\n"):
        line = block.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        events.append(data if data == "[DONE]" else json.loads(data))
    return events
