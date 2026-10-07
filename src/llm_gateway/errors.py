"""Gateway errors, rendered in the OpenAI error envelope so existing SDKs
surface them sensibly."""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    status_code: int = 500
    error_type: str = "server_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_type is not None:
            self.error_type = error_type
        self.code = code
        self.headers = headers or {}

    def to_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        }


class AuthenticationError(GatewayError):
    status_code = 401
    error_type = "authentication_error"


class RateLimitedError(GatewayError):
    status_code = 429
    error_type = "rate_limit_error"


class ModelNotFoundError(GatewayError):
    status_code = 404
    error_type = "invalid_request_error"


class InvalidRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"


class UpstreamError(GatewayError):
    """Every provider in the fallback chain failed."""

    status_code = 502
    error_type = "upstream_error"
