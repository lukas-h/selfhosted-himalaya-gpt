from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] | str | None = None
    seed: int | None = None
    user: str | None = None


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "himalaya-ai"


class ModelsList(BaseModel):
    object: str = "list"
    data: list[ModelObject]


class JobSpec(BaseModel):
    job_id: str
    model: str
    request: dict[str, Any]
    issued_at: float
    deadline_at: float = Field(description="absolute unix time after which master will give up")
