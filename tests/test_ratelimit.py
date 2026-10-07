from __future__ import annotations

from tests.conftest import Harness

TINY = "sk-tiny-key-123456"


async def test_bucket_allows_burst_then_429(harness: Harness) -> None:
    assert (await harness.chat("a", key=TINY)).status_code == 200
    second = await harness.chat("b", key=TINY)
    assert second.status_code == 200
    assert second.headers["x-ratelimit-remaining-requests"] == "0"
    third = await harness.chat("c", key=TINY)
    assert third.status_code == 429
    assert third.json()["error"]["type"] == "rate_limit_error"
    assert int(third.headers["retry-after"]) >= 1


async def test_bucket_refills_over_time(harness: Harness) -> None:
    for content in ("a", "b"):
        await harness.chat(content, key=TINY)
    assert (await harness.chat("c", key=TINY)).status_code == 429
    harness.clock.advance(1.0)  # refill_per_s = 1
    assert (await harness.chat("d", key=TINY)).status_code == 200


async def test_limits_are_per_key(harness: Harness) -> None:
    for content in ("a", "b", "c"):
        await harness.chat(content, key=TINY)
    # A different key has its own bucket.
    assert (await harness.chat("d")).status_code == 200


async def test_rejected_requests_never_reach_providers(harness: Harness) -> None:
    for content in ("a", "b", "c", "d"):
        await harness.chat(content, key=TINY)
    assert harness.providers["primary"].calls == 2
