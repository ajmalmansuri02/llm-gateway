"""Google Gemini via the native Generative Language REST API.

Gemini also offers an OpenAI-compatible endpoint (which the
``openai_compatible`` provider can use directly). This adapter talks to the
native API on purpose: translating between wire formats is a large part of what
real gateways do, and it is worth having written once.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.providers.base import (
    EmbeddingResult,
    Provider,
    ProviderError,
    estimate_tokens,
    map_httpx_error,
    new_completion_id,
    now_ts,
)
from llm_gateway.schemas import ChatCompletionRequest

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
}


def to_gemini_request(request: ChatCompletionRequest) -> dict[str, Any]:
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    for message in request.messages:
        text = message.text()
        if message.role in ("system", "developer"):
            system_parts.append({"text": text})
            continue
        role = "model" if message.role == "assistant" else "user"
        # Gemini requires alternating turns; merge consecutive same-role messages.
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": role, "parts": [{"text": text}]})

    generation: dict[str, Any] = {}
    if request.temperature is not None:
        generation["temperature"] = request.temperature
    if request.top_p is not None:
        generation["topP"] = request.top_p
    max_tokens = request.max_completion_tokens or request.max_tokens
    if max_tokens is not None:
        generation["maxOutputTokens"] = max_tokens
    if request.stop is not None:
        generation["stopSequences"] = (
            [request.stop] if isinstance(request.stop, str) else request.stop
        )

    body: dict[str, Any] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}
    if generation:
        body["generationConfig"] = generation
    return body


def _candidate_text(payload: dict[str, Any]) -> tuple[str, str | None]:
    candidates = payload.get("candidates") or []
    if not candidates:
        return "", None
    first = candidates[0]
    parts = (first.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    reason = first.get("finishReason")
    return text, _FINISH_REASONS.get(reason, "stop") if reason else None


def _usage(payload: dict[str, Any]) -> dict[str, int] | None:
    meta = payload.get("usageMetadata")
    if not meta:
        return None
    prompt = int(meta.get("promptTokenCount", 0))
    completion = int(meta.get("candidatesTokenCount", 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(meta.get("totalTokenCount", prompt + completion)),
    }


def from_gemini_response(payload: dict[str, Any], model: str) -> dict[str, Any]:
    text, finish = _candidate_text(payload)
    result: dict[str, Any] = {
        "id": new_completion_id(),
        "object": "chat.completion",
        "created": now_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish or "stop",
            }
        ],
    }
    usage = _usage(payload)
    if usage:
        result["usage"] = usage
    return result


class GeminiProvider(Provider):
    def __init__(
        self,
        name: str,
        api_key: str | None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, timeout_s)
        self._has_key = bool(api_key)
        self._client = httpx.AsyncClient(
            base_url=(base_url or DEFAULT_BASE_URL).rstrip("/"),
            headers={"x-goog-api-key": api_key or ""},
            timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)),
            transport=transport,
        )

    def _require_key(self) -> None:
        # Surfaces as a fallback-able error instead of a confusing 403 upstream.
        if not self._has_key:
            raise ProviderError("GEMINI_API_KEY is not set", provider=self.name, status_code=401)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self._require_key()
        try:
            response = await self._client.post(path, json=body)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise map_httpx_error(self.name, exc) from exc

    async def chat(self, model: str, request: ChatCompletionRequest) -> dict[str, Any]:
        payload = await self._post(f"/models/{model}:generateContent", to_gemini_request(request))
        return from_gemini_response(payload, model)

    async def stream_chat(
        self, model: str, request: ChatCompletionRequest
    ) -> AsyncIterator[dict[str, Any]]:
        self._require_key()
        completion_id, created = new_completion_id(), now_ts()
        first = True
        try:
            async with self._client.stream(
                "POST",
                f"/models/{model}:streamGenerateContent",
                params={"alt": "sse"},
                json=to_gemini_request(request),
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        payload = json.loads(line[5:].strip())
                    except ValueError as exc:
                        raise ProviderError(
                            "bad SSE payload", provider=self.name, kind="decode"
                        ) from exc
                    text, finish = _candidate_text(payload)
                    delta: dict[str, Any] = {"content": text}
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    chunk: dict[str, Any] = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    usage = _usage(payload)
                    if usage and finish:
                        chunk["usage"] = usage
                    yield chunk
        except httpx.HTTPError as exc:
            raise map_httpx_error(self.name, exc) from exc

    async def embed(self, model: str, inputs: list[str]) -> EmbeddingResult:
        body = {
            "requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": text}]}}
                for text in inputs
            ]
        }
        payload = await self._post(f"/models/{model}:batchEmbedContents", body)
        vectors = [e["values"] for e in payload.get("embeddings", [])]
        # The embed API does not return token counts; estimate them.
        return EmbeddingResult(
            vectors=vectors, prompt_tokens=sum(estimate_tokens(t) for t in inputs)
        )

    async def aclose(self) -> None:
        await self._client.aclose()
