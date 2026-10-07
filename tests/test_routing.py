from __future__ import annotations

from llm_gateway.providers import FakeProvider
from tests.conftest import MakeHarness


async def test_retries_transient_error_on_same_provider(make_harness: MakeHarness) -> None:
    h = await make_harness({"primary": FakeProvider("primary", fail_status=503, fail_times=1)})
    resp = await h.chat("hi")
    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "primary"
    assert h.providers["primary"].calls == 2
    assert len(h.sleeps) == 1  # one backoff between the two attempts


async def test_falls_back_after_retries_exhausted(make_harness: MakeHarness) -> None:
    h = await make_harness({"primary": FakeProvider("primary", fail_status=503)})
    resp = await h.chat("hi")
    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "secondary"
    assert resp.headers["x-gateway-fallbacks"] == "1"
    assert h.providers["primary"].calls == 2  # max_attempts
    assert h.providers["secondary"].calls == 1


async def test_non_retryable_error_falls_back_without_retry(make_harness: MakeHarness) -> None:
    h = await make_harness({"primary": FakeProvider("primary", fail_status=401)})
    resp = await h.chat("hi")
    assert resp.status_code == 200
    assert h.providers["primary"].calls == 1
    assert h.sleeps == []


async def test_client_fault_is_returned_without_fallback(make_harness: MakeHarness) -> None:
    h = await make_harness({"primary": FakeProvider("primary", fail_status=400)})
    resp = await h.chat("hi")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "upstream_rejected"
    assert h.providers["secondary"].calls == 0


async def test_timeout_triggers_fallback(make_harness: MakeHarness) -> None:
    slow = FakeProvider("primary", latency_s=5, timeout_s=0.05)
    h = await make_harness({"primary": slow})
    resp = await h.chat("hi")
    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "secondary"


async def test_all_failures_give_502(make_harness: MakeHarness) -> None:
    h = await make_harness(
        {
            "primary": FakeProvider("primary", fail_kind="connection"),
            "secondary": FakeProvider("secondary", fail_status=502),
        }
    )
    resp = await h.chat("hi")
    assert resp.status_code == 502
    message = resp.json()["error"]["message"]
    assert "[primary] connection" in message
    assert "[secondary] upstream status=502" in message


async def test_backoff_is_exponential_and_capped(make_harness: MakeHarness) -> None:
    h = await make_harness(
        {"primary": FakeProvider("primary", fail_status=503, fail_times=3)},
        retry={"max_attempts": 4, "base_delay_s": 1.0, "max_delay_s": 3.0},
    )
    resp = await h.chat("hi")
    assert resp.status_code == 200
    # nominal delays 1, 2, 4->3 (capped), each scaled by jitter in [0.5, 1.0]
    assert len(h.sleeps) == 3
    for slept, nominal in zip(h.sleeps, [1.0, 2.0, 3.0], strict=True):
        assert nominal * 0.5 <= slept <= nominal
