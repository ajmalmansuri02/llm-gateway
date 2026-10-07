from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_gateway.config import GatewayConfig, expand_env, load_gateway_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_expand_env_with_defaults() -> None:
    env = {"SET": "value", "EMPTY": ""}
    assert expand_env("${SET}", env) == "value"
    assert expand_env("${MISSING:-fallback}", env) == "fallback"
    assert expand_env("${EMPTY:-fallback}", env) == "fallback"
    assert expand_env("${MISSING}", env) == ""
    assert expand_env({"a": ["x-${SET}"]}, env) == {"a": ["x-value"]}


def test_shipped_config_is_valid() -> None:
    config = load_gateway_config(REPO_ROOT / "config" / "gateway.yaml", env={})
    assert "chat-default" in config.models
    assert config.models["embed-default"].kind == "embedding"
    assert config.price_for("ollama", "anything").input_per_1m == 0
    assert config.auth.keys[0].key == "sk-gw-dev-local-only"


def test_shipped_config_reads_env() -> None:
    config = load_gateway_config(
        REPO_ROOT / "config" / "gateway.yaml",
        env={"GATEWAY_DEV_KEY": "sk-from-env-123", "SEMANTIC_CACHE_ENABLED": "true"},
    )
    assert config.auth.keys[0].key == "sk-from-env-123"
    assert config.cache.semantic.enabled is True


def test_unknown_provider_reference_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown provider 'nope'"):
        GatewayConfig.model_validate(
            {
                "providers": {"a": {"type": "fake"}},
                "models": {"m": {"targets": [{"provider": "nope", "model": "x"}]}},
            }
        )


def test_semantic_cache_needs_embedding_alias() -> None:
    with pytest.raises(ValidationError, match="embedding_model"):
        GatewayConfig.model_validate(
            {
                "providers": {"a": {"type": "fake"}},
                "models": {"m": {"targets": [{"provider": "a", "model": "x"}]}},
                "cache": {"semantic": {"enabled": True, "embedding_model": "m"}},
            }
        )


def test_price_lookup_prefers_exact_then_wildcard() -> None:
    config = GatewayConfig.model_validate(
        {
            "providers": {"a": {"type": "fake"}},
            "models": {"m": {"targets": [{"provider": "a", "model": "x"}]}},
            "pricing": {"a/x": {"input_per_1m": 1}, "a/*": {"input_per_1m": 9}},
        }
    )
    assert config.price_for("a", "x").input_per_1m == 1
    assert config.price_for("a", "y").input_per_1m == 9
    assert config.price_for("b", "y").input_per_1m == 0
