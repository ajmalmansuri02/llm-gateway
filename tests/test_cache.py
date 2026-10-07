from __future__ import annotations

from llm_gateway.cache import exact_cache_key
from llm_gateway.schemas import ChatCompletionRequest
from tests.conftest import Harness, MakeHarness


async def test_identical_request_hits_exact_cache(harness: Harness) -> None:
    first = await harness.chat("what is redis?", temperature=0)
    second = await harness.chat("what is redis?", temperature=0)
    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "hit"
    assert second.json()["choices"] == first.json()["choices"]
    assert second.json()["id"] != first.json()["id"]
    assert harness.providers["primary"].calls == 1


async def test_different_params_miss(harness: Harness) -> None:
    await harness.chat("what is redis?", temperature=0)
    resp = await harness.chat("what is redis?", temperature=1)
    assert resp.headers["x-gateway-cache"] == "miss"


async def test_cache_control_no_cache_bypasses(harness: Harness) -> None:
    await harness.chat("bypass me")
    resp = await harness.client.post(
        "/v1/chat/completions",
        json={"model": "chat", "messages": [{"role": "user", "content": "bypass me"}]},
        headers={"authorization": "Bearer sk-test-key-123456", "cache-control": "no-cache"},
    )
    assert resp.headers["x-gateway-cache"] == "bypass"
    assert harness.providers["primary"].calls == 2


async def test_cache_disabled_globally(make_harness: MakeHarness) -> None:
    h = await make_harness(cache={"enabled": False})
    await h.chat("x")
    resp = await h.chat("x")
    assert resp.headers["x-gateway-cache"] == "bypass"


def test_cache_key_ignores_stream_and_user() -> None:
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    a = ChatCompletionRequest.model_validate(base)
    b = ChatCompletionRequest.model_validate({**base, "stream": True, "user": "bob"})
    c = ChatCompletionRequest.model_validate({**base, "temperature": 0.5})
    assert exact_cache_key("alias", a) == exact_cache_key("alias", b)
    assert exact_cache_key("alias", a) != exact_cache_key("alias", c)
    assert exact_cache_key("alias", a) != exact_cache_key("other", a)


async def test_semantic_cache_hits_on_similar_prompt(harness: Harness) -> None:
    first = await harness.chat("How do I reset my password", model="chat-sem")
    assert first.headers["x-gateway-cache"] == "miss"
    # Same words, different casing/punctuation: different exact key, same meaning.
    second = await harness.chat("how do i reset my password?", model="chat-sem")
    assert second.headers["x-gateway-cache"] == "semantic-hit"
    assert harness.providers["primary"].calls == 1


async def test_semantic_cache_misses_on_unrelated_prompt(harness: Harness) -> None:
    await harness.chat("How do I reset my password", model="chat-sem")
    resp = await harness.chat("Explain token bucket rate limiting", model="chat-sem")
    assert resp.headers["x-gateway-cache"] == "miss"


async def test_semantic_cache_is_opt_in_per_alias(harness: Harness) -> None:
    await harness.chat("How do I reset my password")
    resp = await harness.chat("how do i reset my password?")
    assert resp.headers["x-gateway-cache"] == "miss"
