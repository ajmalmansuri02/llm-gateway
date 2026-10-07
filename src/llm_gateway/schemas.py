"""Request models for the OpenAI-compatible surface.

Only the fields the gateway itself reasons about are typed. Everything else is
kept (``extra="allow"``) and passed through to OpenAI-compatible upstreams, so
clients can use features the gateway does not know about yet.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    # A string, or a list of content parts ({"type": "text", "text": ...}, ...).
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(
            str(part.get("text", "")) for part in self.content if part.get("type") == "text"
        )


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    user: str | None = None

    def upstream_payload(self, model: str, *, stream: bool) -> dict[str, Any]:
        """The body to send to an OpenAI-compatible upstream."""
        payload = self.model_dump(exclude_none=True, exclude={"stream_options"})
        payload["model"] = model
        payload["stream"] = stream
        return payload

    def prompt_text(self) -> str:
        return "\n".join(f"{m.role}: {m.text()}" for m in self.messages)


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[str]

    def inputs(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)
