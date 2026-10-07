from __future__ import annotations

from tests.conftest import Harness


async def test_non_streaming_returns_openai_shape(harness: Harness) -> None:
    resp = await harness.chat("hello world")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "chat"  # the alias the client asked for
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "echo: hello world"}
    assert body["usage"]["total_tokens"] > 0
    assert resp.headers["x-gateway-provider"] == "primary"
    assert resp.headers["x-gateway-upstream-model"] == "p-model"
    assert resp.headers["x-gateway-cache"] == "miss"
    assert "x-request-id" in resp.headers


async def test_missing_api_key_is_401(harness: Harness) -> None:
    resp = await harness.client.post(
        "/v1/chat/completions",
        json={"model": "chat", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_api_key"


async def test_wrong_api_key_is_401(harness: Harness) -> None:
    resp = await harness.chat(key="sk-nope-nope-nope")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


async def test_unknown_model_is_404(harness: Harness) -> None:
    resp = await harness.chat(model="does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


async def test_embedding_alias_rejected_for_chat(harness: Harness) -> None:
    resp = await harness.chat(model="embed")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "wrong_model_kind"


async def test_direct_provider_model_addressing(harness: Harness) -> None:
    resp = await harness.chat(model="secondary/some-model")
    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "secondary"
    assert resp.headers["x-gateway-upstream-model"] == "some-model"


async def test_validation_errors_use_openai_envelope(harness: Harness) -> None:
    resp = await harness.client.post(
        "/v1/chat/completions",
        json={"model": "chat", "messages": []},
        headers={"authorization": "Bearer sk-test-key-123456"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


async def test_list_models(harness: Harness) -> None:
    resp = await harness.client.get(
        "/v1/models", headers={"authorization": "Bearer sk-test-key-123456"}
    )
    ids = {m["id"] for m in resp.json()["data"]}
    assert ids == {"chat", "chat-sem", "embed"}


async def test_embeddings(harness: Harness) -> None:
    resp = await harness.client.post(
        "/v1/embeddings",
        json={"model": "embed", "input": ["a cat", "a dog"]},
        headers={"authorization": "Bearer sk-test-key-123456"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert len(body["data"][0]["embedding"]) == 64
    assert body["usage"]["prompt_tokens"] > 0
