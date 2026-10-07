from __future__ import annotations

import logging

import pytest
from prometheus_client import REGISTRY

from llm_gateway.redaction import redact, redact_obj
from tests.conftest import Harness, MakeHarness


def sample(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def test_healthz(harness: Harness) -> None:
    resp = await harness.client.get("/healthz")
    assert resp.json() == {"status": "ok"}


async def test_readyz_checks_redis(harness: Harness) -> None:
    resp = await harness.client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["checks"]["redis"] == "ok"


async def test_readyz_reports_redis_down(harness: Harness) -> None:
    async def broken_ping() -> bool:
        raise ConnectionError("down")

    harness.state.redis.ping = broken_ping  # type: ignore[method-assign]
    resp = await harness.client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


async def test_metrics_endpoint_exposes_gateway_metrics(harness: Harness) -> None:
    hits_before = sample("gateway_cache_lookups_total", {"cache": "exact", "result": "hit"})
    await harness.chat("metric me")
    await harness.chat("metric me")
    body = (await harness.client.get("/metrics")).text
    for name in (
        "gateway_request_duration_seconds_bucket",
        "gateway_provider_requests_total",
        "gateway_tokens_total",
        "gateway_cache_lookups_total",
    ):
        assert name in body
    assert sample("gateway_cache_lookups_total", {"cache": "exact", "result": "hit"}) == (
        hits_before + 1
    )


async def test_provider_errors_are_counted(make_harness: MakeHarness) -> None:
    from llm_gateway.providers import FakeProvider

    labels = {"provider": "primary", "kind": "upstream", "status": "503"}
    before = sample("gateway_provider_errors_total", labels)
    h = await make_harness({"primary": FakeProvider("primary", fail_status=503)})
    await h.chat("hi")
    assert sample("gateway_provider_errors_total", labels) == before + 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail me at jane.doe@example.com", "mail me at [REDACTED_EMAIL]"),
        ("card 4111 1111 1111 1111 ok", "card [REDACTED_CARD] ok"),
        ("ssn 123-45-6789", "ssn [REDACTED_SSN]"),
        ("call +91 98765 43210 now", "call [REDACTED_PHONE] now"),
        ("aadhaar 2345 6789 0123", "aadhaar [REDACTED_AADHAAR]"),
        ("pan ABCDE1234F", "pan [REDACTED_PAN]"),
        ("key sk-abcdefghijklmnop1234", "key [REDACTED_SECRET]"),
        ("from 10.0.0.12", "from [REDACTED_IP]"),
        ("nothing to see here", "nothing to see here"),
    ],
)
def test_redaction(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_card_requires_luhn() -> None:
    # 16 digits that fail the Luhn check are left alone by the card rule.
    assert "[REDACTED_CARD]" not in redact("order 1234 5678 9012 3456")


def test_redact_obj_walks_nested_structures() -> None:
    data = {"messages": [{"role": "user", "content": "I am bob@example.org"}], "n": 1}
    assert redact_obj(data) == {
        "messages": [{"role": "user", "content": "I am [REDACTED_EMAIL]"}],
        "n": 1,
    }


async def test_body_logging_is_redacted(
    make_harness: MakeHarness, caplog: pytest.LogCaptureFixture
) -> None:
    h = await make_harness(logging={"log_bodies": True, "redact_pii": True})
    with caplog.at_level(logging.INFO, logger="llm_gateway.requests"):
        await h.chat("my email is alice@example.com")
    record = next(r for r in caplog.records if r.getMessage() == "chat completion")
    assert "alice@example.com" not in str(record.__dict__)
    assert "[REDACTED_EMAIL]" in str(record.messages)  # type: ignore[attr-defined]
    assert record.api_key == "tester"  # type: ignore[attr-defined]


async def test_bodies_not_logged_by_default(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="llm_gateway.requests"):
        await harness.chat("secret stuff")
    record = next(r for r in caplog.records if r.getMessage() == "chat completion")
    assert not hasattr(record, "messages")
