# -*- coding: utf-8 -*-
"""测试共用 fixture：全部跑在离线替身 + 不落盘埋点的配置上，CI 无需任何密钥。"""

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.core.config import LLMConfig, MetricsConfig, ServerConfig, Settings


@pytest.fixture()
def offline_settings() -> Settings:
    return Settings(
        env="test",
        version="0.1.0",
        server=ServerConfig(),
        llm=LLMConfig(provider="mock", model="echo-1"),
        metrics=MetricsConfig(enabled=False),
        prompts_dir="config/prompts",
    )


@pytest.fixture()
def client(offline_settings: Settings):
    with TestClient(create_app(offline_settings)) as c:
        yield c
