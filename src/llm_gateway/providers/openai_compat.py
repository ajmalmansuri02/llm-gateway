"""Any server that speaks the OpenAI REST API: Ollama (``/v1``), vLLM, llama.cpp
server, LM Studio, LiteLLM, OpenAI itself, Gemini's OpenAI-compatible endpoint..."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.providers.base import (
    EmbeddingResult,
    Provider,
    ProviderError,
    map_httpx_error,
)
from llm_gateway.schemas import ChatCompletionRequest


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        *,
        stream_usage: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, timeout_s)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.stream_usage = stream_usage
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)),
            transport=transport,
        )

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise map_httpx_error(self.name, exc) from exc

    async def chat(self, model: str, request: ChatCompletionRequest) -> dict[str, Any]:
        return await self._post_json(
            "/chat/completions", request.upstream_payload(model, stream=False)
        )

    async def stream_chat(
        self, model: str, request: ChatCompletionRequest
    ) -> AsyncIterator[dict[str, Any]]:
        payload = request.upstream_payload(model, stream=True)
        if self.stream_usage:
            payload["stream_options"] = {"include_usage": True}
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except ValueError as exc:
                        raise ProviderError(
                            f"bad SSE payload: {data[:200]}", provider=self.name, kind="decode"
                        ) from exc
        except httpx.HTTPError as exc:
            raise map_httpx_error(self.name, exc) from exc

    async def embed(self, model: str, inputs: list[str]) -> EmbeddingResult:
        body = await self._post_json("/embeddings", {"model": model, "input": inputs})
        data = sorted(body.get("data", []), key=lambda d: d.get("index", 0))
        usage = body.get("usage") or {}
        return EmbeddingResult(
            vectors=[d["embedding"] for d in data],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
