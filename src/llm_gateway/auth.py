"""API key authentication.

Clients send ``Authorization: Bearer <key>`` exactly as they would to OpenAI. The
key *name* (never the key itself) is what shows up in logs, metrics and usage.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from llm_gateway.config import AuthConfig, RateLimitConfig
from llm_gateway.errors import AuthenticationError


@dataclass(frozen=True)
class Principal:
    name: str
    rate_limit: RateLimitConfig


class Authenticator:
    def __init__(self, config: AuthConfig) -> None:
        self._config = config

    def authenticate(self, authorization: str | None) -> Principal:
        if not self._config.enabled:
            return Principal("anonymous", self._config.default_rate_limit)
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthenticationError(
                "Missing API key. Send 'Authorization: Bearer <key>'.", code="missing_api_key"
            )
        token = token.strip().encode()
        match = None
        # Compare against every key so timing does not reveal which prefix matched.
        for entry in self._config.keys:
            if hmac.compare_digest(token, entry.key.encode()):
                match = entry
        if match is None:
            raise AuthenticationError("Invalid API key.", code="invalid_api_key")
        return Principal(match.name, match.rate_limit or self._config.default_rate_limit)
