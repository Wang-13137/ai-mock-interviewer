# -*- coding: utf-8 -*-
"""评测中台的数据契约。

这个模块里最重要的字段不是指标，而是 `trusted`。

原因：本项目有一条硬红线——**替身（EchoClient / HashingEmbedder）跑出来的数字不得外引**。
过去这条红线只写在 README 和文档里，靠人记；一旦有人拿替身跑出的 0.83 去找工作，
就是把文档承诺变成了谎话。

中台把红线编码进数据结构：
    · 每个 run 落库时判定 trusted（真实模型 + 真实嵌入才算 true）；
    · gate 只允许 trusted run 之间比较——拿替身当基线，等于用一个假分数当尺子；
    · 不可信的 run 依然可以跑、可以看，只是永远进不了「可对外引用」的口径。

这样"不许引用替身数字"从人的自觉变成了系统的约束，这是中台相对脚本的本质区别。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvalTask(str, Enum):
    """评测任务类型。新增任务只需在 tasks.py 注册一个函数。"""

    RETRIEVAL = "retrieval"      # Recall@k / MRR@10 / Hit@1，零 LLM，离线可跑
    GENERATION = "generation"    # LLM-as-judge 五维评分 + 与人工标注的一致性
    AGENT = "agent"              # 工具调用率 / 完成率 / 平均步数


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"        # 前置条件不满足，拒绝产出数字（而不是产出假数字）


class Verdict(str, Enum):
    PASS = "pass"
    REGRESSED = "regressed"
    INCOMPARABLE = "incomparable"  # 配置/数据/可信度不一致，不允许比
    NOT_GATED = "not_gated"        # 没有给基线，本次只登记不判定


class EvalRunRequest(BaseModel):
    """触发一次评测。"""

    task: EvalTask
    # 数据集中台里的 dataset id；留空则用任务默认数据集
    dataset_id: str = ""
    # 任务参数（例如 generation 的 limit / concurrency，retrieval 的 top_k）
    params: Dict[str, Any] = Field(default_factory=dict)
    # 回归基线；给了才判 pass/regressed
    baseline_run: Optional[str] = None


class CaseRow(BaseModel):
    """逐条明细：没有明细的评测只有结论、没有归因能力。"""

    case_id: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    detail: Dict[str, Any] = Field(default_factory=dict)


class MetricDelta(BaseModel):
    metric: str
    baseline: float
    current: float
    delta: float
    delta_pct: float
    higher_is_better: bool = True
    regressed: bool = False


class GateResult(BaseModel):
    """回归门禁结论。"""

    verdict: Verdict
    baseline_run: str = ""
    threshold: float = 0.02
    deltas: List[MetricDelta] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)

    @property
    def regressed_metrics(self) -> List[MetricDelta]:
        return [d for d in self.deltas if d.regressed]


class EvalRunRecord(BaseModel):
    """一次评测运行的完整记录（落库 + 对外返回）。"""

    id: str
    task: str
    dataset_id: str = ""
    dataset_hash: str = ""
    status: RunStatus = RunStatus.SUCCEEDED
    verdict: Verdict = Verdict.NOT_GATED
    # 是否由真实模型 / 真实嵌入产出——决定它能不能被引用
    trusted: bool = False
    # 配置指纹：provider/model/strategy 等算 hash，只有指纹相同才可比
    config_fingerprint: str = ""
    fingerprint_detail: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    gate: Optional[GateResult] = None
    warnings: List[str] = Field(default_factory=list)
    error: str = ""
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str = ""
    duration_ms: float = 0.0
    cases: int = 0

    def summary_line(self) -> str:
        tag = "可信" if self.trusted else "不可引用"
        main = self.metrics.get("mrr@10") or self.metrics.get("mean_score") or 0.0
        return f"{self.id} {self.task:<10} [{tag}] main={main} verdict={self.verdict.value}"


class TaskInfo(BaseModel):
    """可用任务的元信息（给 /v1/eval/tasks 与 CLI 用）。"""

    name: str
    title: str
    default_dataset: str
    requires_llm: bool
    metrics: List[str] = Field(default_factory=list)
    description: str = ""
