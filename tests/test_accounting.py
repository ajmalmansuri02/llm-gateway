from __future__ import annotations

import pytest

from llm_gateway.providers import FakeProvider
from tests.conftest import API_KEY, Harness, MakeHarness

AUTH = {"authorization": f"Bearer {API_KEY}"}


async def test_usage_and_cost_from_price_table(make_harness: MakeHarness) -> None:
    # 400 chars -> 100 prompt tokens ("user: " prefix adds a little), fixed reply.
    h = await make_harness({"primary": FakeProvider("primary", reply="x" * 800)})
    resp = await h.chat("y" * 400)
    usage = resp.json()["usage"]
    expected = (usage["prompt_tokens"] * 1.0 + usage["completion_tokens"] * 2.0) / 1_000_000

    totals = (await h.client.get("/v1/usage", headers=AUTH)).json()
    assert totals["requests"] == 1
    assert totals["completion_tokens"] == 200
    assert totals["prompt_tokens"] == usage["prompt_tokens"]
    assert totals["cost_usd"] == pytest.approx(expected)
    assert totals["tokens_by_model"] == {"primary/p-model": usage["total_tokens"]}


async def test_wildcard_price_applies_to_fallback_provider(make_harness: MakeHarness) -> None:
    h = await make_harness(
        {
            "primary": FakeProvider("primary", fail_status=503),
            "secondary": FakeProvider("secondary", reply="z" * 40),
        }
    )
    resp = await h.chat("hi")
    usage = resp.json()["usage"]
    totals = (await h.client.get("/v1/usage", headers=AUTH)).json()
    expected = (usage["prompt_tokens"] * 10 + usage["completion_tokens"] * 20) / 1_000_000
    assert totals["cost_usd"] == pytest.approx(expected)


async def test_cache_hits_are_free(harness: Harness) -> None:
    await harness.chat("same")
    await harness.chat("same")
    totals = (await harness.client.get("/v1/usage", headers=AUTH)).json()
    assert totals["requests"] == 2
    assert totals["cache_hits"] == 1
    first_only = (await harness.chat("same")).json()["usage"]
    assert totals["prompt_tokens"] == first_only["prompt_tokens"]


async def test_usage_is_isolated_per_key(harness: Harness) -> None:
    await harness.chat("hi")
    other = await harness.client.get(
        "/v1/usage", headers={"authorization": "Bearer sk-tiny-key-123456"}
    )
    assert other.json()["requests"] == 0
