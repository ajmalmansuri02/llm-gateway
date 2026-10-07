"""Provider adapters and the factory that builds them from config."""

from __future__ import annotations

from llm_gateway.config import ProviderConfig
from llm_gateway.providers.base import EmbeddingResult, Provider, ProviderError
from llm_gateway.providers.fake import FakeProvider
from llm_gateway.providers.gemini import GeminiProvider
from llm_gateway.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "EmbeddingResult",
    "FakeProvider",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "build_provider",
]


def build_provider(name: str, cfg: ProviderConfig) -> Provider:
    if cfg.type == "openai_compatible":
        if not cfg.base_url:
            raise ValueError(f"provider '{name}' needs base_url")
        return OpenAICompatibleProvider(
            name,
            cfg.base_url,
            cfg.api_key or None,
            cfg.timeout_s,
            stream_usage=bool(cfg.options.get("stream_usage", True)),
        )
    if cfg.type == "gemini":
        return GeminiProvider(name, cfg.api_key or None, cfg.base_url, cfg.timeout_s)
    if cfg.type == "fake":
        return FakeProvider.from_options(name, cfg.options, cfg.timeout_s)
    raise ValueError(f"unknown provider type {cfg.type!r}")  # pragma: no cover
