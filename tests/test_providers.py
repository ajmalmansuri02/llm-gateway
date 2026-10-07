"""Provider adapters against mocked HTTP (httpx.MockTransport)."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_gateway.providers import GeminiProvider, OpenAICompatibleProvider, ProviderError
from llm_gateway.providers.gemini import to_gemini_request
from llm_gateway.schemas import ChatCompletionRequest


def chat_request(**extra: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "alias",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            **extra,
        }
    )


# ---------------------------------------------------------------- OpenAI-compatible
async def test_openai_compat_chat_passes_through_and_rewrites_model() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        "p", "http://up/v1", api_key="k", transport=httpx.MockTransport(handler)
    )
    result = await provider.chat("real-model", chat_request(temperature=0.2, seed=7))
    assert result["choices"][0]["message"]["content"] == "ok"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "real-model"
    assert body["stream"] is False
    assert body["seed"] == 7  # unknown fields pass through


async def test_openai_compat_stream_parses_sse() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
        ": keep-alive comment\n\n"
        'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream_options"] == {"include_usage": True}
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    provider = OpenAICompatibleProvider("p", "http://up/v1", transport=httpx.MockTransport(handler))
    chunks = [c async for c in provider.stream_chat("m", chat_request())]
    assert [c["choices"][0]["delta"]["content"] for c in chunks] == ["he", "llo"]


@pytest.mark.parametrize(
    ("status", "retryable", "fallback"),
    [(503, True, True), (429, True, True), (401, False, True), (400, False, False)],
)
async def test_openai_compat_maps_http_errors(status: int, retryable: bool, fallback: bool) -> None:
    provider = OpenAICompatibleProvider(
        "p",
        "http://up/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(status, text="nope")),
    )
    with pytest.raises(ProviderError) as info:
        await provider.chat("m", chat_request())
    assert info.value.status_code == status
    assert info.value.retryable is retryable
    assert info.value.should_fallback is fallback


async def test_openai_compat_connection_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = OpenAICompatibleProvider("p", "http://up/v1", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as info:
        await provider.chat("m", chat_request())
    assert info.value.kind == "connection"
    assert info.value.retryable


async def test_openai_compat_embeddings_sorted_by_index() -> None:
    payload = {
        "data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}],
        "usage": {"prompt_tokens": 3},
    }
    provider = OpenAICompatibleProvider(
        "p",
        "http://up/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)),
    )
    result = await provider.embed("e", ["a", "b"])
    assert result.vectors == [[0.1], [0.2]]
    assert result.prompt_tokens == 3


# ---------------------------------------------------------------------------- Gemini
def test_gemini_request_translation() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": [{"type": "text", "text": "q2"}]},
                {"role": "user", "content": "q3"},
            ],
            "temperature": 0.3,
            "max_tokens": 50,
            "stop": "END",
        }
    )
    body = to_gemini_request(req)
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][2]["parts"] == [{"text": "q2"}, {"text": "q3"}]  # merged turns
    assert body["generationConfig"] == {
        "temperature": 0.3,
        "maxOutputTokens": 50,
        "stopSequences": ["END"],
    }


async def test_gemini_chat_translates_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gem:generateContent"
        assert request.headers["x-goog-api-key"] == "key"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hel"}, {"text": "lo"}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
            },
        )

    provider = GeminiProvider("g", "key", transport=httpx.MockTransport(handler))
    result = await provider.chat("gem", chat_request())
    assert result["choices"][0]["message"]["content"] == "Hello"
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


async def test_gemini_stream_translates_chunks() -> None:
    events = [
        {"candidates": [{"content": {"parts": [{"text": "Hi "}]}}]},
        {
            "candidates": [{"content": {"parts": [{"text": "there"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
        },
    ]
    sse = "".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["alt"] == "sse"
        return httpx.Response(200, text=sse)

    provider = GeminiProvider("g", "key", transport=httpx.MockTransport(handler))
    chunks = [c async for c in provider.stream_chat("gem", chat_request())]
    assert chunks[0]["choices"][0]["delta"] == {"content": "Hi ", "role": "assistant"}
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[1]["usage"]["total_tokens"] == 5
    assert len({c["id"] for c in chunks}) == 1


async def test_gemini_without_key_fails_fast_and_falls_back() -> None:
    provider = GeminiProvider(
        "g", None, transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    with pytest.raises(ProviderError) as info:
        await provider.chat("gem", chat_request())
    assert info.value.status_code == 401
    assert info.value.should_fallback
    assert not info.value.retryable


async def test_gemini_embeddings() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert len(body["requests"]) == 2
        return httpx.Response(200, json={"embeddings": [{"values": [1.0]}, {"values": [2.0]}]})

    provider = GeminiProvider("g", "key", transport=httpx.MockTransport(handler))
    result = await provider.embed("text-embedding", ["a", "b"])
    assert result.vectors == [[1.0], [2.0]]
