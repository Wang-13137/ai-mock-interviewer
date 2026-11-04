# -*- coding: utf-8 -*-
"""集中配置读取：YAML 默认值 + 环境变量覆盖 + 厂商预设解析。

环境变量规则：``AIMI_`` 前缀，双下划线分层。
    AIMI_LLM__PROVIDER=zhipu       -> {"llm": {"provider": "zhipu"}}
    AIMI_LLM__API_KEY=sk-xxx       -> {"llm": {"api_key": "sk-xxx"}}
    AIMI_METRICS__ENABLED=false    -> {"metrics": {"enabled": False}}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

from ..version import __version__

ENV_PREFIX = "AIMI_"
ENV_DELIM = "__"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


# --------------------------------------------------------------------------- #
# 配置模型
# --------------------------------------------------------------------------- #
class LLMConfig(BaseModel):
    provider: str = "mock"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 30.0
    max_retries: int = 2


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class MetricsConfig(BaseModel):
    enabled: bool = True
    path: str = "logs/metrics.jsonl"


class AgentConfig(BaseModel):
    """工具调用（Agent Loop）相关配置。"""

    enabled: bool = True
    max_steps: int = 3
    # 工具调用协议：native（默认，把工具以 OpenAI tools 参数下发）| handwritten（```tool 代码块塞进 system）
    protocol: str = "native"
    question_bank: str = "config/question_bank.json"


class MemoryConfig(BaseModel):
    """W5 会话记忆：历史估算 token 超预算后滚动压缩成摘要（见 agent/memory.py）。"""

    enabled: bool = True
    max_tokens: int = 3000
    keep_recent: int = 6
    summary_max_chars: int = 600


class RagConfig(BaseModel):
    """检索配置。

    `strategy` 三选一：bm25（零依赖）/ vector / hybrid（后两者需要 sentence-transformers，
    缺失时自动退化为 HashingEmbedder 替身，并在 /health 与报告里显式标记）。
    """

    strategy: str = "bm25"
    embedder: str = "auto"  # auto | hashing | sentence-transformers
    # Hub id 或本地路径；Hub id 会优先解析为 models/<basename> 本地镜像（见 vector._resolve_model_ref）
    model_name: str = "BAAI/bge-small-zh-v1.5"
    # 用 bge-small-zh-v1.5 在 80 语料 × 60 条评测集上标定（2026-10-16 二轮）：
    # 平台区 0.30–0.40（Recall@3=0.800 / MRR@10=0.817 / semantic 子集 0.634），取上沿 0.40；
    # 0.45 起缓慢下滑，0.60 崩（semantic 0.408）。换模型/扩语料后重跑 scripts/calibrate_min_score.py
    min_score: float = 0.40
    # cross-encoder 精排：none（默认关闭，粗排够用且零开销）| cross-encoder（需本地权重）
    rerank: str = "none"
    reranker_model: str = "BAAI/bge-reranker-base"


class DataHubConfig(BaseModel):
    """数据中台：数据集登记、质量门禁、版本与血缘。

    `db_path` 只存元数据（JSON 数据文件本身仍是唯一的事实源）。
    """

    enabled: bool = True
    db_path: str = "data/datahub.db"


class EvalHubConfig(BaseModel):
    """评测中台：任务编排、指标落库、与基线的回归门禁。

    `gate_threshold` 是回归判定的默认阈值（相对跌幅，0.02 = 2 个百分点）。
    设太小会天天报回归（噪声），设太大则真退步也发现不了；0.02 是在本项目
    检索指标的量级（0.80 上下、样本 60 条）上，噪声与真实差异的实用分界。
    """

    enabled: bool = True
    db_path: str = "data/evalhub.db"
    gate_threshold: float = 0.02


class Settings(BaseModel):
    env: str = "dev"
    version: str = __version__
    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # 裁判必须用与生成不同的模型，否则是自评偏差
    judge: LLMConfig = Field(default_factory=LLMConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    datahub: DataHubConfig = Field(default_factory=DataHubConfig)
    evalhub: EvalHubConfig = Field(default_factory=EvalHubConfig)
    prompts_dir: str = "config/prompts"


# --------------------------------------------------------------------------- #
# 厂商预设：全部走 OpenAI 兼容的 /chat/completions 协议
# --------------------------------------------------------------------------- #
PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    # key_env = 未显式填 api_key 时，去读哪个环境变量
    "mock": {"base_url": "", "model": "echo-1", "key_env": ""},
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",  # 免费额度，开发阶段首选
        "key_env": "ZHIPU_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "key_env": "SILICONFLOW_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
}


class ConfigError(RuntimeError):
    """配置不可用（例如 provider 未登记且没给 base_url）。"""


# --------------------------------------------------------------------------- #
# 加载与合并
# --------------------------------------------------------------------------- #
def _coerce(raw: str) -> Any:
    """环境变量都是字符串，按 YAML 的习惯还原成标量类型。"""
    low = raw.strip().lower()
    if low in ("true", "yes", "1"):
        return True
    if low in ("false", "no", "0"):
        return False
    if low in ("null", "none", ""):
        return None
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            pass
    return raw


def _deep_set(data: Dict[str, Any], keys: list[str], value: Any) -> None:
    cur: Any = data
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _env_overrides() -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for name, raw in os.environ.items():
        if not name.startswith(ENV_PREFIX) or ENV_DELIM not in name:
            continue
        parts = [p.lower() for p in name[len(ENV_PREFIX):].split(ENV_DELIM) if p]
        if len(parts) < 2:
            continue
        _deep_set(merged, parts, _coerce(raw))
    return merged


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取（比引入 python-dotenv 更轻，够用）。不覆盖已存在的环境变量。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        os.environ.setdefault(key, val)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_settings(
    config_path: Optional[str | Path] = None,
    dotenv: str | Path = ".env",
    **overrides: Any,
) -> Settings:
    """读取配置：YAML 兜底 -> 环境变量覆盖 -> 显式 overrides 最高。"""
    _load_dotenv(Path(dotenv))

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    data: Dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    data = _deep_merge(data, _env_overrides())
    data = _deep_merge(data, overrides)
    return Settings(**data)


def resolve_llm(cfg: LLMConfig) -> LLMConfig:
    """把厂商预设补进空字段：base_url / model / api_key。"""
    if cfg.provider not in PROVIDER_PRESETS and not cfg.base_url:
        raise ConfigError(
            f"未登记的 provider: {cfg.provider!r}；"
            f"请显式提供 llm.base_url，或改用 {sorted(PROVIDER_PRESETS)}"
        )

    preset = PROVIDER_PRESETS.get(cfg.provider, {})
    out = cfg.model_copy(deep=True)

    out.base_url = (cfg.base_url or preset.get("base_url", "")).rstrip("/")
    out.model = cfg.model or preset.get("model", "")

    if not out.api_key:
        key_env = preset.get("key_env", "")
        if key_env:
            out.api_key = os.environ.get(key_env, "")
        if not out.api_key:
            out.api_key = os.environ.get("AIMI_LLM__API_KEY", "")
    return out


def is_ready(cfg: LLMConfig) -> bool:
    """是否具备调用真实模型的最小条件（provider 非 mock 且有 Key 与地址）。"""
    try:
        resolved = resolve_llm(cfg)
    except ConfigError:
        return False
    return bool(resolved.provider != "mock" and resolved.api_key and resolved.base_url)
