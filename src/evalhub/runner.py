# -*- coding: utf-8 -*-
"""评测中台的编排层：一次 run 从绑定数据快照到落库的完整生命周期。

一次 run 的固定顺序（顺序本身就是设计）：
    1. 解析任务与数据集；
    2. **绑定数据快照**（content_hash）——先固定"用什么数据"，再跑；
       反过来（先跑再记数据）就会出现"指标有、出处没有"的情况；
    3. 自动补同步：数据集未登记或已 dirty 时先 sync 一次，避免拿不一致的数据跑；
    4. 执行任务，得到 metrics + 逐条明细；
    5. 合成 trusted：任务自证可信 且 数据可信，两者缺一不可；
    6. 算配置指纹（**不含代码版本**，否则门禁就抓不到代码改动）；
    7. 有基线则跑门禁（带逐条明细 → 配对 bootstrap 判显著性）；
    8. 落库（run + 明细），失败也要落库——失败记录本身是排障依据。

两种执行模式（v0.8 起）：
    · `run()`   同步跑完再返回。CLI 与测试用，语义简单，错误直接抛给调用方；
    · `submit()` 立刻返回 PENDING 记录，后台任务继续跑。HTTP 用——
      生成评测要几分钟，同步接口在网关 60s 超时后必然断连，
      而 run 还在跑、库里却没有它，就成了**幽灵 run**。

`baseline_run="latest"` 是一个实用约定：自动取同一任务下最近一个 trusted run 作基线，
这样 CI 里不需要人工记 run id 就能做回归检测。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

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
from .provenance import code_version, runtime_snapshot
from .store import EvalHubStore
from .tasks import TASKS, TaskContext, list_tasks

log = logging.getLogger("aimi.evalhub")

# 阶段名会写进 run 记录，轮询方据此判断"还在跑"还是"卡住了"
STAGE_QUEUED = "queued"
STAGE_BINDING = "binding_dataset"
STAGE_RUNNING = "running_task"
STAGE_GATING = "gating"


def config_fingerprint(detail: Dict[str, Any]) -> str:
    """配置指纹：把影响结果的配置（模型/策略/阈值…）算成一个短 hash。

    只有指纹相同才允许比较——否则「换了模型导致指标变化」会被误判成「代码退步」。
    **刻意不含 git sha 与依赖版本**：那两样一变就判"不可比"，门禁就永远检测不到
    代码改动造成的退步，而这正是它存在的理由。溯源信息单独记录在 `code_version`。
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
        *,
        max_concurrency: Optional[int] = None,
        run_timeout_s: Optional[float] = None,
        bootstrap_iters: Optional[int] = None,
    ) -> None:
        self.settings = settings
        self.datahub = datahub
        self.store = EvalHubStore(db_path or settings.evalhub.db_path)
        cfg = settings.evalhub
        self.gate_threshold = cfg.gate_threshold if gate_threshold is None else gate_threshold
        self.gate_abs_threshold = cfg.gate_abs_threshold
        self.max_concurrency = cfg.max_concurrency if max_concurrency is None else max_concurrency
        self.run_timeout_s = cfg.run_timeout_s if run_timeout_s is None else run_timeout_s
        self.bootstrap_iters = cfg.bootstrap_iters if bootstrap_iters is None else bootstrap_iters
        # 事件循环相关对象惰性创建：Semaphore 绑定 loop，跨 loop 复用会报错
        self._sem: Optional[asyncio.Semaphore] = None
        self._sem_loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks: Set[asyncio.Task] = set()

    # ---------------- 查询 ---------------- #
    def tasks(self) -> List[TaskInfo]:
        return list_tasks()

    def get_run(self, run_id: str, with_cases: bool = False) -> Optional[EvalRunRecord]:
        return self.store.get_run(run_id, with_cases=with_cases)

    def list_runs(self, task: Optional[str] = None, limit: int = 20, only_trusted: bool = False) -> List[EvalRunRecord]:
        return self.store.list_runs(task=task, limit=limit, only_trusted=only_trusted)

    def get_cases(self, run_id: str, limit: Optional[int] = None):
        return self.store.list_cases(run_id, limit=limit)

    def prune(self, keep_last: int = 50, keep_trusted: int = 20, dry_run: bool = False) -> Dict[str, Any]:
        return self.store.prune(keep_last=keep_last, keep_trusted=keep_trusted, dry_run=dry_run)

    # ---------------- 数据快照 ---------------- #
    def _prepare_dataset(self, dataset_id: str) -> Dict[str, Any]:
        snap = self.datahub.snapshot(dataset_id)
        # 未登记或磁盘已变：先补一次同步，再取快照，避免拿不一致的数据跑
        if not snap.get("registered") or snap.get("dirty"):
            self.datahub.sync(dataset_id)
            snap = self.datahub.snapshot(dataset_id)
        return snap

    # ---------------- run 创建 ---------------- #
    def _new_record(self, req: EvalRunRequest, run_id: Optional[str], status: RunStatus) -> EvalRunRecord:
        return EvalRunRecord(
            id=run_id or _new_run_id(),
            task=req.task.value,
            status=status,
            verdict=Verdict.NOT_GATED,
            started_at=datetime.now().isoformat(timespec="seconds"),
            # 溯源信息在创建时写一次，后续不覆盖
            code_version=code_version(),
            runtime=dict(runtime_snapshot()),
            stage=STAGE_QUEUED if not status.is_terminal else "",
        )

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(max(1, self.max_concurrency))
            self._sem_loop = loop
        return self._sem

    async def submit(self, req: EvalRunRequest) -> EvalRunRecord:
        """异步提交：立刻落库并返回 PENDING，执行交给后台任务。

        HTTP 层用这个。返回后调用方轮询 `GET /runs/{id}` 看 status/stage/progress。
        """
        record = self._new_record(req, None, RunStatus.PENDING)
        self.store.save_run(record)
        task = asyncio.create_task(self._guarded(record, req), name=f"evalhub-{record.id}")
        # 必须持有引用：否则任务可能被 GC 回收（asyncio 的经典坑）
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        log.info("evalhub 提交 run %s task=%s", record.id, record.task)
        return record

    async def _guarded(self, record: EvalRunRecord, req: EvalRunRequest) -> None:
        """后台执行外壳：并发闸门 + 超时兜底 + 任何异常都要落库。"""
        async with self._semaphore():
            try:
                await asyncio.wait_for(self._execute(record, req), timeout=self.run_timeout_s)
            except asyncio.TimeoutError:
                record.status = RunStatus.FAILED
                record.error = f"执行超时（>{self.run_timeout_s:.0f}s），已中止并落库"
                record.stage = ""
                record.finished_at = datetime.now().isoformat(timespec="seconds")
                self.store.save_run(record)
                log.error("evalhub run %s 超时", record.id)
            except Exception as exc:  # 兜底：后台任务的异常没人 await，不落库就等于静默丢失
                log.exception("evalhub 后台 run %s 异常", record.id)
                record.status = RunStatus.FAILED
                record.error = f"{type(exc).__name__}: {exc}"
                record.stage = ""
                record.finished_at = datetime.now().isoformat(timespec="seconds")
                self.store.save_run(record)

    async def wait(self, run_id: str, timeout: float = 600.0, interval: float = 0.05) -> Optional[EvalRunRecord]:
        """轮询到终态。给 CLI 和测试用——HTTP 客户端应自己轮询而不是占着连接。"""
        deadline = time.monotonic() + timeout
        while True:
            record = self.store.get_run(run_id)
            if record is None or record.status.is_terminal:
                return record
            if time.monotonic() >= deadline:
                return record
            await asyncio.sleep(interval)

    # ---------------- 执行 ---------------- #
    async def run(self, req: EvalRunRequest, run_id: Optional[str] = None) -> EvalRunRecord:
        """同步执行一次 run（阻塞直到终态）。CLI / 测试路径。"""
        record = self._new_record(req, run_id, RunStatus.PENDING)
        self.store.save_run(record)
        async with self._semaphore():
            await self._execute(record, req)
        return record

    async def _execute(self, record: EvalRunRecord, req: EvalRunRequest) -> None:
        started = time.perf_counter()
        task_def = TASKS.get(req.task.value)

        if task_def is None:
            self._reject(record, f"未注册的任务类型：{req.task.value}")
            return

        record.status = RunStatus.RUNNING
        record.stage = STAGE_BINDING
        record.progress = 0.05
        self.store.save_run(record)

        dataset_id = req.dataset_id or task_def.info.default_dataset
        snap = await asyncio.to_thread(self._prepare_dataset, dataset_id)
        if not snap.get("registered"):
            self._reject(
                record,
                f"数据集 {dataset_id} 未登记且同步失败：先修数据质量问题"
                f"（质量门禁见 /v1/data/datasets/{dataset_id}）",
                dataset_id=dataset_id,
            )
            return

        record.dataset_id = dataset_id
        record.dataset_hash = snap.get("content_hash", "")
        if not snap.get("trusted_data"):
            record.warnings.append(f"数据集 {dataset_id} 质量状态为 {snap.get('quality_status')}，结果需谨慎解读")
        record.stage = STAGE_RUNNING
        record.progress = 0.15
        self.store.save_run(record)

        ctx = TaskContext(settings=self.settings, datahub=self.datahub)
        try:
            result = await task_def.runner(ctx, dataset_id, dict(req.params))
        except Exception as exc:  # 任务崩了也要落库，否则无法复盘
            log.exception("评测任务异常 %s", req.task.value)
            record.status = RunStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            record.stage = ""
            record.finished_at = datetime.now().isoformat(timespec="seconds")
            record.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            self.store.save_run(record)
            return

        if result.error:
            record.status = RunStatus.FAILED
            record.error = result.error
            record.metrics = result.metrics
            record.warnings.extend(result.warnings)
            record.stage = ""
            record.finished_at = datetime.now().isoformat(timespec="seconds")
            record.duration_ms = round((time.perf_counter() - started) * 1000, 1)
            self.store.save_run(record)
            return

        record.metrics = result.metrics
        record.warnings.extend(result.warnings)
        record.fingerprint_detail = result.fingerprint_detail
        record.config_fingerprint = config_fingerprint(result.fingerprint_detail)
        # 双条件：任务自证可信 + 数据可信
        record.trusted = bool(result.trusted and snap.get("trusted_data"))
        record.status = RunStatus.SUCCEEDED
        record.stage = STAGE_GATING if req.baseline_run else ""
        record.progress = 0.85 if req.baseline_run else 1.0
        record.finished_at = datetime.now().isoformat(timespec="seconds")
        record.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        record.cases = len(result.cases)
        self.store.save_run(record)
        self.store.save_cases(record.id, result.cases)

        if req.baseline_run:
            gate = self.gate_run(record.id, baseline=req.baseline_run)
            if gate is not None:
                record.gate = gate
                record.verdict = gate.verdict
                self.store.set_verdict(record.id, gate)
            record.stage = ""
            record.progress = 1.0
            self.store.save_run(record)

        log.info(
            "evalhub run %s task=%s trusted=%s verdict=%s code=%s",
            record.id, record.task, record.trusted, record.verdict.value, record.code_version or "?",
        )

    def _reject(self, record: EvalRunRecord, reason: str, dataset_id: str = "") -> None:
        """拒绝产出数字。**宁可没有结果，也不要一个来源不明的结果。**"""
        record.status = RunStatus.REJECTED
        record.verdict = Verdict.INCOMPARABLE
        record.error = reason
        record.stage = ""
        if dataset_id:
            record.dataset_id = dataset_id
        record.finished_at = datetime.now().isoformat(timespec="seconds")
        self.store.save_run(record)
        log.warning("evalhub 拒绝运行 %s：%s", record.id, reason)

    # ---------------- 门禁 ---------------- #
    def gate_run(
        self,
        run_id: str,
        baseline: Optional[str] = None,
        threshold: Optional[float] = None,
        *,
        abs_threshold: Optional[float] = None,
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
                abs_threshold=self.gate_abs_threshold,
                reasons=["没有可用的基线 run（同任务、trusted）"],
            )

        base = self.store.get_run(baseline_id)
        if base is None:
            return GateResult(
                verdict=Verdict.INCOMPARABLE,
                baseline_run=baseline_id,
                threshold=self.gate_threshold,
                abs_threshold=self.gate_abs_threshold,
                reasons=[f"基线 run 不存在：{baseline_id}"],
            )

        # 逐条明细做配对 bootstrap：这是"点估计退步"与"统计上真的退步"的分界
        base_cases = self.store.list_cases(baseline_id)
        cur_cases = self.store.list_cases(run_id)

        gate = compare(
            base,
            current,
            pct_threshold=threshold if threshold is not None else self.gate_threshold,
            abs_threshold=self.gate_abs_threshold if abs_threshold is None else abs_threshold,
            base_cases=base_cases or None,
            cur_cases=cur_cases or None,
            iters=self.bootstrap_iters,
        )
        self.store.set_verdict(run_id, gate)
        return gate

    async def aclose(self) -> None:
        """关停时取消仍在跑的后台任务，避免进程退出时留下半截状态。"""
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(BaseException):
                await task
        self._tasks.clear()
