# -*- coding: utf-8 -*-
"""请求模型的校验规则测试。"""

import pytest
from pydantic import ValidationError

from src.schemas.chat import ChatMessage, ChatRequest, InterviewProfile


def test_last_message_must_be_user():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[ChatMessage(role="user", content="hi"),
                              ChatMessage(role="assistant", content="hello")])


def test_empty_messages_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_empty_content_rejected():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


def test_defaults_and_turn_count():
    req = ChatRequest(messages=[ChatMessage(role="user", content="你好")])
    assert req.session_id == "anonymous"
    assert req.profile.position == "AI 应用工程师"
    assert req.turn_count == 1


def test_temperature_bounds():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[ChatMessage(role="user", content="hi")], temperature=5.0)


def test_profile_customization():
    p = InterviewProfile(position="算法工程师", style="压力面试", company="某某科技")
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], profile=p)
    assert req.profile.style == "压力面试"
    assert req.profile.company == "某某科技"


def test_invalid_style_rejected():
    with pytest.raises(ValidationError):
        InterviewProfile(style="随便聊聊")
