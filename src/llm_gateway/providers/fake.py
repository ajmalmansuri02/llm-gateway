"""A deterministic in-process provider.

Used by the test-suite and for load testing the gateway itself without a GPU:
it can echo, add latency, and fail on demand, which makes retries, fallbacks and
timeouts easy to observe.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import AsyncIterator
from typing import Any

from llm_gateway.providers.base import (
    EmbeddingResult,
    Provider,
    ProviderError,
    estimate_tokens,
    new_completion_id,
    now_ts,
)
from llm_gateway.schemas import ChatCompletionRequest

_WORD = re.compile(r"[a-z0-9]+")


def bag_of_words_vector(text: str, dim: int = 64) -> list[float]:
    """Hash words into a fixed-size unit vector. Texts that share words end up
    close in cosine space, which is enough to exercise the semantic cache."""
    vec = [0.0] * dim
    for word in _WORD.findall(text.lower()):
        index = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16) % dim
        vec[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeProvider(Provider):
    def __init__(
        self,
        name: str = "fake",
        *,
        reply: str | None = None,
        latency_s: float = 0.0,
        fail_status: int | None = None,
        fail_times: int | None = None,
        fail_kind: str = "upstream",
        chunk_words: bool = True,
        embedding_dim: int = 64,
        timeout_s: float = 30.0,
    ) -> None:
        super().__init__(name, timeout_s)
        self.reply = reply
        self.latency_s = latency_s
        self.fail_status = fail_status
        # None -> fail forever (when fail_status/kind set); N -> fail the first N calls.
        self.fail_times = fail_times
        self.fail_kind = fail_kind
        self.chunk_words = chunk_words
        self.embedding_dim = embedding_dim
        self.calls = 0

    @classmethod
    def from_options(cls, name: str, options: dict[str, Any], timeout_s: float) -> FakeProvider:
        return cls(name, timeout_s=timeout_s, **options)

    def _should_fail(self) -> bool:
        if self.fail_status is None and self.fail_kind == "upstream":
            return False
        return self.fail_times is None or self.calls <= self.fail_times

    async def _before_call(self) -> None:
        self.calls += 1
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self._should_fail():
            raise ProviderError(
                "injected failure",
                provider=self.name,
                status_code=self.fail_status,
                kind=self.fail_kind,
            )

    def _reply_for(self, request: ChatCompletionRequest) -> str:
        if self.reply is not None:
            return self.reply
        return f"echo: {request.messages[-1].text()}"

    async def chat(self, model: str, request: ChatCompletionRequest) -> dict[str, Any]:
        await self._before_call()
        text = self._reply_for(request)
        prompt_tokens = estimate_tokens(request.prompt_text())
        completion_tokens = estimate_tokens(text)
        return {
            "id": new_completion_id(),
            "object": "chat.completion",
            "created": now_ts(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def stream_chat(
        self, model: str, request: ChatCompletionRequest
    ) -> AsyncIterator[dict[str, Any]]:
        await self._before_call()
        text = self._reply_for(request)
        pieces = [w + " " for w in text.split(" ")] if self.chunk_words else [text]
        pieces[-1] = pieces[-1].rstrip(" ")
        completion_id, created = new_completion_id(), now_ts()
        for i, piece in enumerate(pieces):
            delta: dict[str, Any] = {"content": piece}
            if i == 0:
                delta["role"] = "assistant"
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        prompt_tokens = estimate_tokens(request.prompt_text())
        completion_tokens = estimate_tokens(text)
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def embed(self, model: str, inputs: list[str]) -> EmbeddingResult:
        await self._before_call()
        return EmbeddingResult(
            vectors=[bag_of_words_vector(t, self.embedding_dim) for t in inputs],
            prompt_tokens=sum(estimate_tokens(t) for t in inputs),
        )
