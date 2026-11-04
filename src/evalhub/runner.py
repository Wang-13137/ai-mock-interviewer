# -*- coding: utf-8 -*-
"""评测中台的编排层：一次 run 从绑定数据快照到落库的完整生命周期。

一次 run 的固定顺序（顺序本身就是设计）：
    1. 解析任务与数据集；
    2. **绑定数据快照**（content_hash）——先固定"用什么数据"，再跑；
       反过来（先跑再记数据）就会出现"指标有、出处没有"的情况；
    3. 自动补同步：数据集未登记或已 dirty 时先 sync 一次，避免拿不一致的数据跑；
    4. 执行任务，得到 metrics + 逐条明细；
    5. 合成 trusted：任务自证可信 且 数据可信，两者缺一不可；
    6. 算配置指纹；
    7. 有基线则跑门禁；
    8. 落库（run + 明细），失败也要落库——失败记录本身是排障依据。

`baseline_run="latest"` 是一个实用约定：自动取同一任务下最近一个 trusted run 作基线，
这样 CI 里不需要人工记 run id 就能做回归检测。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.config import Settings
from .gate import compare
from .models import (
    EvalRunRecord,
    EvalRunRequest,
    GateResult,
    RunStatus,
    TaskInfo,
    Verdict,
)
from .store import EvalHubStore
from .tasks import TASKS, TaskContext, list_tasks

log = logging.getLogger("aimi.evalhub")


def config_fingerprint(detail: Dict[str, Any]) -> str:
    """配置指纹：把影响结果的配置（模型/策略/阈值…）算成一个短 hash。

    只有指纹相同才允许比较——否则「换了模型导致指标变化」会被误判成「代码退步」。
    """
    payload = json.dumps(detail, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"r-{stamp}-{uuid.uuid4().hex[:4]}"


class EvalHub:
    """评测任务编排器。"""

    def __init__(
        self,
        settings: Settings,
        datahub: Any,
        db_path: Optional[str] = None,
        gate_threshold: Optional[float] = None,
    ) -> None:
        self.settings = settings
        self.datahub = datahub
        self.store = EvalHubStore(db_path or settings.evalhub.db_path)
        self.gate_threshold = settings.evalhub.gate_threshold if gate_threshold is None else gate_threshold

    # ---------------- 查询 ---------------- #
    def tasks(self) -> List[TaskInfo]:
        return list_tasks()

    def get_run(self, run_id: str, with_cases: bool = False) -> Optional[EvalRunRecord]:
        return self.store.get_run(run_id, with_cases=with_cases)

    def list_runs(self, task: Optional[str] = None, limit: int = 20, only_trusted: bool = False) -> List[EvalRunRecord]:
        return self.store.list_runs(task=task, limit=limit, only_trusted=only_trusted)

    def get_cases(self, run_id: str, limit: Optional[int] = None):
        return self.store.list_cases(run_id, limit=limit)

    # ---------------- 数据快照 ---------------- #
    def _prepare_dataset(self, dataset_id: str) -> Dict[str, Any]:
        snap = self.datahub.snapshot(dataset_id)
        # 未登记或磁盘已变：先补一次同步，再取快照，避免拿不一致的数据跑
        if not snap.get("registered") or snap.get("dirty"):
            self.datahub.sync(dataset_id)
            snap = self.datahub.snapshot(dataset_id)
        return snap

    # ---------------- 执行 ---------------- #
    async def run(self, req: EvalRunRequest, run_id: Optional[str] = None) -> EvalRunRecord:
        task_def = TASKS.get(req.task.value)
        run_id = run_id or _new_run_id()
        started = time.perf_counter()

        if task_def is None:
            return self._reject(run_id, req, f"未注册的任务类型：{req.task.value}")

        dataset_id = req.dataset_id or task_def.info.default_dataset
        snap = self._prepare_dataset(dataset_id)
        if not snap.get("registered"):
            return self._reject(
                run_id,
                req,
                f"数据集 {dataset_id} 未登记且同步失败：先修数据质量问题（质量门禁见 /v1/data/datasets/{dataset_id}）",
                dataset_id=dataset_id,
            )

        record = EvalRunRecord(
            id=run_id,
            task=req.task.value,
            dataset_id=dataset_id,
            dataset_hash=snap.get("content_hash", ""),
            status=RunStatus.SUCCEEDED,
            verdict=Verdict.NOT_GATED,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        if not snap.get("trusted_data"):
            record.warnings.append(f"数据集 {dataset_id} 质量状态为 {snap.get('quality_status')}，结果需谨慎解读")

        ctx = TaskContext(settings=self.settings, datahub=self.datahub)
        try:
            result = await task_def.runner(ctx, dataset_id, dict(req.params))
        except Exception as exc:  # 任务崩了也要落库，否则无法复盘
            log.exception("评测任务异常 %s", req.task.value)
            record.status = RunStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            record.finished_at = datetime.now().isoformat(timespec="seconds")
            record.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            self.store.save_run(record)
            return record

        if result.error:
            record.status = RunStatus.FAILED
            record.error = result.error
            record.metrics = result.metrics
            record.warnings.extend(result.warnings)
            record.finished_at = datetime.now().isoformat(timespec="seconds")
            record.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            self.store.save_run(record)
            return record

        record.metrics = result.metrics
        record.warnings.extend(result.warnings)
        record.fingerprint_detail = result.fingerprint_detail
        record.config_fingerprint = config_fingerprint(result.fingerprint_detail)
        # 双条件：任务自证可信 + 数据可信
        record.trusted = bool(result.trusted and snap.get("trusted_data"))
        record.status = RunStatus.SUCCEEDED
        record.finished_at = datetime.now().isoformat(timespec="seconds")
        record.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        record.cases = len(result.cases)
        self.store.save_run(record)
        self.store.save_cases(run_id, result.cases)

        if req.baseline_run:
            gate = self.gate_run(run_id, baseline=req.baseline_run)
            if gate is not None:
                record.gate = gate
                record.verdict = gate.verdict
                self.store.set_verdict(run_id, gate)

        log.info("evalhub run %s task=%s trusted=%s verdict=%s", run_id, record.task, record.trusted, record.verdict.value)
        return record

    def _reject(self, run_id: str, req: EvalRunRequest, reason: str, dataset_id: str = "") -> EvalRunRecord:
        """拒绝产出数字。**宁可没有结果，也不要一个来源不明的结果。**"""
        record = EvalRunRecord(
            id=run_id,
            task=req.task.value,
            dataset_id=dataset_id or req.dataset_id,
            status=RunStatus.REJECTED,
            verdict=Verdict.INCOMPARABLE,
            error=reason,
            started_at=datetime.now().isoformat(timespec="seconds"),
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.store.save_run(record)
        log.warning("evalhub 拒绝运行 %s：%s", run_id, reason)
        return record

    # ---------------- 门禁 ---------------- #
    def gate_run(
        self,
        run_id: str,
        baseline: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Optional[GateResult]:
        current = self.store.get_run(run_id)
        if current is None:
            return None

        baseline_id = baseline
        if baseline_id == "latest" or baseline_id is None:
            candidates = [
                r
                for r in self.store.list_runs(task=current.task, limit=50, only_trusted=True)
                if r.id != current.id
            ]
            baseline_id = candidates[0].id if candidates else None
        if not baseline_id:
            return GateResult(
                verdict=Verdict.NOT_GATED,
                threshold=self.gate_threshold,
                reasons=["没有可用的基线 run（同任务、trusted）"],
            )

        base = self.store.get_run(baseline_id)
        if base is None:
            return GateResult(
                verdict=Verdict.INCOMPARABLE,
                baseline_run=baseline_id,
                threshold=self.gate_threshold,
                reasons=[f"基线 run 不存在：{baseline_id}"],
            )

        gate = compare(base, current, threshold if threshold is not None else self.gate_threshold)
        self.store.set_verdict(run_id, gate)
        return gate
