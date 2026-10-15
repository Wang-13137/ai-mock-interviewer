# -*- coding: utf-8 -*-
"""配置加载与厂商预设解析的测试。"""

import os

import pytest

from src.core.config import ConfigError, LLMConfig, PROVIDER_PRESETS, is_ready, load_settings, resolve_llm


def test_yaml_defaults_are_mock():
    s = load_settings(dotenv=".env.not-exist")
    assert s.llm.provider in PROVIDER_PRESETS
    assert s.server.port == 8000


def test_env_overrides_yaml(monkeypatch, tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("llm:\n  provider: mock\n  temperature: 0.7\n", encoding="utf-8")

    monkeypatch.setenv("AIMI_LLM__PROVIDER", "deepseek")
    monkeypatch.setenv("AIMI_LLM__TEMPERATURE", "0.1")
    monkeypatch.setenv("AIMI_METRICS__ENABLED", "false")

    s = load_settings(yaml_path, dotenv=".env.not-exist")
    assert s.llm.provider == "deepseek"
    assert s.llm.temperature == 0.1
    assert s.metrics.enabled is False


def test_resolve_llm_fills_preset():
    cfg = resolve_llm(LLMConfig(provider="deepseek"))
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.model == "deepseek-chat"


def test_resolve_llm_reads_vendor_key_from_env(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "dummy-key")
    cfg = resolve_llm(LLMConfig(provider="zhipu"))
    assert cfg.api_key == "dummy-key"
    assert cfg.base_url.endswith("/api/paas/v4")


def test_resolve_llm_rejects_unknown_provider_without_base_url():
    with pytest.raises(ConfigError):
        resolve_llm(LLMConfig(provider="no-such-vendor"))


def test_readiness_requires_key_and_url(monkeypatch):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("AIMI_LLM__API_KEY", raising=False)
    assert is_ready(LLMConfig(provider="mock")) is False
    assert is_ready(LLMConfig(provider="zhipu")) is False
    monkeypatch.setenv("ZHIPU_API_KEY", "dummy")
    assert is_ready(LLMConfig(provider="zhipu")) is True
