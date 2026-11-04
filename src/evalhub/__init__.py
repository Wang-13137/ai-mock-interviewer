# -*- coding: utf-8 -*-
"""评测中台：把一次性评测脚本变成可编排、可追溯、可回归的流水线。

模块结构：
    models.py   数据契约（Run / Gate / MetricDelta / TaskInfo）
    store.py    SQLite（run 元数据 + 逐条明细）
    tasks.py    三类任务实现（retrieval / generation / agent）
    gate.py     回归门禁：可比性检查 + 指标差集
    runner.py   编排：快照绑定 -> 执行 -> trusted 合成 -> 门禁 -> 落库
    api.py      REST 路由（挂在 /v1/eval/*）

最短用法：
    from src.evalhub import EvalHub, EvalRunRequest, EvalTask
    hub = EvalHub(load_settings(), datahub)
    rec = await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL, baseline_run="latest"))
    print(rec.metrics, rec.verdict)
"""

from .gate import compare, higher_is_better, render_gate
from .models import (
    CaseRow,
    EvalRunRecord,
    EvalRunRequest,
    EvalTask,
    GateResult,
    MetricDelta,
    RunStatus,
    TaskInfo,
    Verdict,
)
from .runner import EvalHub, config_fingerprint
from .store import EvalHubStore
from .tasks import TASKS, TaskContext, TaskResult, list_tasks

__all__ = [
    "CaseRow",
    "EvalHub",
    "EvalHubStore",
    "EvalRunRecord",
    "EvalRunRequest",
    "EvalTask",
    "GateResult",
    "MetricDelta",
    "RunStatus",
    "TASKS",
    "TaskContext",
    "TaskInfo",
    "TaskResult",
    "Verdict",
    "compare",
    "config_fingerprint",
    "higher_is_better",
    "list_tasks",
    "render_gate",
]
