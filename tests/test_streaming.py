from __future__ import annotations

from llm_gateway.providers import FakeProvider
from tests.conftest import Harness, MakeHarness, parse_sse


async def test_stream_emits_chunks_then_done(harness: Harness) -> None:
    resp = await harness.chat("stream me please", stream=True)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all(c["model"] == "chat" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "echo: stream me please"
    # usage is only sent when asked for
    assert not any("usage" in c for c in chunks)


async def test_stream_include_usage(harness: Harness) -> None:
    resp = await harness.chat("count me", stream=True, stream_options={"include_usage": True})
    events = parse_sse(resp.text)
    usage_chunk = events[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["total_tokens"] > 0


async def test_stream_falls_back_before_first_chunk(make_harness: MakeHarness) -> None:
    h = await make_harness({"primary": FakeProvider("primary", fail_status=503)})
    resp = await h.chat("hi", stream=True)
    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "secondary"
    assert resp.headers["x-gateway-fallbacks"] == "1"
    assert parse_sse(resp.text)[-1] == "[DONE]"


async def test_streamed_response_is_cached_and_replayed(harness: Harness) -> None:
    await harness.chat("cache my stream", stream=True)
    resp = await harness.chat("cache my stream", stream=True)
    assert resp.headers["x-gateway-cache"] == "hit"
    events = parse_sse(resp.text)
    assert events[0]["choices"][0]["delta"]["content"] == "echo: cache my stream"
    assert harness.providers["primary"].calls == 1


async def test_stream_all_providers_fail_returns_json_error(make_harness: MakeHarness) -> None:
    h = await make_harness(
        {
            "primary": FakeProvider("primary", fail_status=500),
            "secondary": FakeProvider("secondary", fail_status=500),
        }
    )
    resp = await h.chat("hi", stream=True)
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "all_providers_failed"
