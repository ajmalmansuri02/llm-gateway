"""The provider contract.

Every provider speaks the OpenAI wire format *to the gateway*: ``chat`` returns a
``chat.completion`` dict and ``stream_chat`` yields ``chat.completion.chunk`` dicts.
Translation to and from a provider's native API lives inside the provider.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from llm_gateway.schemas import ChatCompletionRequest

# Upstream statuses worth retrying on the same target.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# Upstream statuses that mean "the request itself is bad" - switching provider will
# not help, so the error goes straight back to the client.
CLIENT_FAULT_STATUSES = frozenset({400, 413, 422})


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        kind: str = "upstream",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        # "timeout" | "connection" | "upstream" | "decode"
        self.kind = kind

    @property
    def retryable(self) -> bool:
        if self.kind in ("timeout", "connection"):
            return True
        return self.status_code in RETRYABLE_STATUSES

    @property
    def should_fallback(self) -> bool:
        return self.status_code not in CLIENT_FAULT_STATUSES

    def __str__(self) -> str:
        status = f" status={self.status_code}" if self.status_code else ""
        return f"[{self.provider}] {self.kind}{status}: {self.message}"


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    prompt_tokens: int
    extra: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    def __init__(self, name: str, timeout_s: float = 60.0) -> None:
        self.name = name
        self.timeout_s = timeout_s

    @abstractmethod
    async def chat(self, model: str, request: ChatCompletionRequest) -> dict[str, Any]: ...

    @abstractmethod
    def stream_chat(
        self, model: str, request: ChatCompletionRequest
    ) -> AsyncIterator[dict[str, Any]]: ...

    @abstractmethod
    async def embed(self, model: str, inputs: list[str]) -> EmbeddingResult: ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


def map_httpx_error(provider: str, exc: Exception) -> ProviderError:
    """Normalise transport exceptions into ``ProviderError``."""
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(str(exc) or "timed out", provider=provider, kind="timeout")
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:500] if exc.response is not None else ""
        return ProviderError(
            body or str(exc), provider=provider, status_code=exc.response.status_code
        )
    if isinstance(exc, httpx.TransportError):
        return ProviderError(str(exc) or type(exc).__name__, provider=provider, kind="connection")
    return ProviderError(str(exc), provider=provider, kind="upstream")


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def now_ts() -> int:
    return int(time.time())


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token for English).

    Used only when a provider does not report usage. Good enough for dashboards
    and budgets, not for billing.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
