# -*- coding: utf-8 -*-
"""请求 / 响应的数据契约（Pydantic v2）。

对外 API 只认这里的模型，内部实现可以随便替换。
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Role = Literal["system", "user", "assistant"]

InterviewStyle = Literal["专业严谨", "压力面试", "轻松交流"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(..., min_length=1, max_length=8000)


class InterviewProfile(BaseModel):
    """一场面试的设定。W2 起这里会带上岗位 JD 与题库来源。"""

    position: str = "AI 应用工程师"
    level: str = "校招应届"
    style: InterviewStyle = "专业严谨"
    company: str = ""
    extra: str = ""


class ChatRequest(BaseModel):
    session_id: str = Field(default="anonymous", max_length=64)
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=100)
    profile: InterviewProfile = Field(default_factory=InterviewProfile)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)

    @field_validator("messages")
    @classmethod
    def last_must_be_user(cls, value: List[ChatMessage]) -> List[ChatMessage]:
        if value[-1].role != "user":
            raise ValueError("最后一条消息必须是 user 角色（系统只会接着用户发言继续）")
        return value

    @property
    def turn_count(self) -> int:
        """用户发言轮数，用于记忆管理与评测埋点。"""
        return sum(1 for m in self.messages if m.role == "user")


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)
    trace_id: str = ""
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str = "ok"
    env: str = "dev"
    provider: str = ""
    model: str = ""
    ready: bool = False
    version: str = ""
