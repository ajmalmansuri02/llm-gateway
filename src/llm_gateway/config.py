"""Configuration.

Two layers:

* ``Settings`` - process-level settings from environment variables (where Redis is,
  where the YAML file is, where to send traces).
* ``GatewayConfig`` - the routing table, providers, prices, API keys and feature
  flags, loaded from a YAML file. ``${VAR}`` and ``${VAR:-default}`` are expanded
  from the environment so secrets never have to live in the file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class Settings(BaseSettings):
    """Environment-driven process settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gateway_config_path: str = "config/gateway.yaml"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    log_json: bool = True
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "llm-gateway"


class ProviderConfig(BaseModel):
    type: Literal["openai_compatible", "gemini", "fake"]
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = 60.0
    # Free-form provider-specific options (used by the fake provider, for example).
    options: dict[str, Any] = Field(default_factory=dict)


class TargetConfig(BaseModel):
    provider: str
    model: str
    # Overrides the provider timeout for this target (covers time-to-first-chunk
    # for streams).
    timeout_s: float | None = None


class ModelConfig(BaseModel):
    """An alias that clients use in the ``model`` field, mapped to an ordered
    fallback chain of provider targets."""

    kind: Literal["chat", "embedding"] = "chat"
    targets: list[TargetConfig] = Field(min_length=1)
    cache: bool = True
    semantic_cache: bool = False


class PriceConfig(BaseModel):
    """USD per one million tokens."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0


class RetryConfig(BaseModel):
    max_attempts: int = Field(default=2, ge=1)
    base_delay_s: float = Field(default=0.2, ge=0)
    max_delay_s: float = Field(default=2.0, ge=0)


class RateLimitConfig(BaseModel):
    """Token bucket: ``capacity`` is the burst size, ``refill_per_s`` the
    sustained request rate."""

    capacity: float = Field(default=60, gt=0)
    refill_per_s: float = Field(default=1.0, gt=0)


class ApiKeyConfig(BaseModel):
    name: str
    key: str = Field(min_length=1)
    rate_limit: RateLimitConfig | None = None


class AuthConfig(BaseModel):
    enabled: bool = True
    keys: list[ApiKeyConfig] = Field(default_factory=list)
    default_rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class SemanticCacheConfig(BaseModel):
    enabled: bool = False
    embedding_model: str = "embed-default"
    threshold: float = Field(default=0.95, gt=0, le=1)
    max_entries: int = Field(default=500, gt=0)


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = Field(default=3600, gt=0)
    semantic: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)


class LoggingConfig(BaseModel):
    log_bodies: bool = False
    redact_pii: bool = True


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    pricing: dict[str, PriceConfig] = Field(default_factory=dict)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Allow clients to bypass aliases with "provider/model" (e.g. "ollama/llama3.2:1b").
    allow_direct_models: bool = True

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        for alias, model in self.models.items():
            for target in model.targets:
                if target.provider not in self.providers:
                    raise ValueError(
                        f"model '{alias}' references unknown provider '{target.provider}'"
                    )
        if self.cache.semantic.enabled:
            emb = self.models.get(self.cache.semantic.embedding_model)
            if emb is None or emb.kind != "embedding":
                raise ValueError(
                    "cache.semantic.embedding_model must name a model alias of kind 'embedding'"
                )
        names = [k.name for k in self.auth.keys]
        if len(names) != len(set(names)):
            raise ValueError("API key names must be unique")
        return self

    def price_for(self, provider: str, model: str) -> PriceConfig:
        return self.pricing.get(f"{provider}/{model}") or self.pricing.get(
            f"{provider}/*", PriceConfig()
        )


def expand_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    env = dict(os.environ) if env is None else env
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            found = env.get(name)
            if found:
                return found
            return default if default is not None else ""

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, env) for v in value]
    return value


def load_gateway_config(path: str | Path, env: dict[str, str] | None = None) -> GatewayConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return GatewayConfig.model_validate(expand_env(raw, env))
