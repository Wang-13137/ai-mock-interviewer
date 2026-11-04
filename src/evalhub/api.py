# -*- coding: utf-8 -*-
"""评测中台 REST 路由（/v1/eval/*）。

`POST /runs` 默认**异步**：落库为 pending 后立刻返回 202 + run_id，
执行在后台任务里跑。理由不是"先进"，而是生成类评测要几分钟——
同步接口会在网关 60s 超时后断开，而 run 还在跑、调用方却什么都拿不到。

需要同步结果的场景（脚本、CI 短任务）加 `?wait=true`，行为与旧版一致（返回 200）。
被质量门禁拒绝的 run 会以 `status=rejected` 落库——异步路径下客户端通过轮询看到它，
**而不是靠 HTTP 400 才知道"这次没有数字"**。这是异步化的真实代价，写在这里免得被当 bug。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from ..core.auth import require_api_key
from .models import CaseRow, EvalRunRecord, EvalRunRequest, GateResult, RunStatus, TaskInfo

router = APIRouter(prefix="/v1/eval", tags=["eval"])


def _hub(request: Request):
    hub = getattr(request.app.state, "evalhub", None)
    if hub is None:
        raise HTTPException(status_code=503, detail="评测中台未启用（见 config 的 evalhub.enabled）")
    return hub


@router.get("/tasks", response_model=List[TaskInfo], summary="可用评测任务")
def list_tasks(request: Request) -> List[TaskInfo]:
    return _hub(request).tasks()


@router.post(
    "/runs",
    response_model=EvalRunRecord,
    status_code=202,
    summary="提交一次评测（默认异步，wait=true 则阻塞到终态）",
    dependencies=[Depends(require_api_key)],
)
async def create_run(
    req: EvalRunRequest,
    request: Request,
    response: Response,
    wait: bool = Query(default=False, description="true = 阻塞到终态并返回 200（仅适合秒级任务）"),
) -> EvalRunRecord:
    hub = _hub(request)
    if wait:
        record = await hub.run(req)
        response.status_code = 200
        if record.status is RunStatus.REJECTED:
            # 同步路径仍用 400：拒绝运行是业务结论而非服务故障
            raise HTTPException(status_code=400, detail={"run_id": record.id, "reason": record.error})
        return record

    record = await hub.submit(req)
    response.headers["Location"] = f"/v1/eval/runs/{record.id}"
    response.headers["Retry-After"] = "2"
    return record


@router.get("/runs", response_model=List[EvalRunRecord], summary="历史 run 列表")
def list_runs(
    request: Request,
    task: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=200),
    trusted_only: bool = False,
) -> List[EvalRunRecord]:
    return _hub(request).list_runs(task=task, limit=limit, only_trusted=trusted_only)


@router.get("/runs/{run_id}", response_model=EvalRunRecord, summary="run 详情（含 status/stage/progress）")
def get_run(run_id: str, request: Request, with_cases: bool = False) -> EvalRunRecord:
    record = _hub(request).get_run(run_id, with_cases=with_cases)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run 不存在：{run_id}")
    return record


@router.get("/runs/{run_id}/progress", summary="轻量进度（轮询用，不返回完整记录）")
def get_progress(run_id: str, request: Request) -> Dict[str, Any]:
    record = _hub(request).get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run 不存在：{run_id}")
    return {
        "id": record.id,
        "status": record.status.value,
        "stage": record.stage,
        "progress": record.progress,
        "cases": record.cases,
        "duration_ms": record.duration_ms,
        "verdict": record.verdict.value,
        "error": record.error,
    }


@router.get("/runs/{run_id}/cases", response_model=List[CaseRow], summary="逐条明细（归因用）")
def get_cases(run_id: str, request: Request, limit: Optional[int] = None) -> List[CaseRow]:
    hub = _hub(request)
    if hub.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"run 不存在：{run_id}")
    return hub.get_cases(run_id, limit=limit)


@router.post(
    "/runs/{run_id}/gate",
    response_model=GateResult,
    summary="与基线比较（回归门禁，含配对 bootstrap）",
    dependencies=[Depends(require_api_key)],
)
def gate_run(
    run_id: str,
    request: Request,
    baseline: Optional[str] = Query(default="latest", description="基线 run id，或 latest"),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="相对阈值"),
    abs_threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="绝对阈值"),
) -> GateResult:
    gate = _hub(request).gate_run(run_id, baseline=baseline, threshold=threshold, abs_threshold=abs_threshold)
    if gate is None:
        raise HTTPException(status_code=404, detail=f"run 不存在：{run_id}")
    return gate


@router.post(
    "/prune",
    summary="按任务裁剪历史 run（保留最近 N 条 + 最近 K 个可信基线）",
    dependencies=[Depends(require_api_key)],
)
def prune(
    request: Request,
    keep_last: int = Query(default=50, ge=1),
    keep_trusted: int = Query(default=20, ge=0),
    dry_run: bool = Query(default=False, description="true = 只统计不删除"),
) -> Dict[str, Any]:
    return _hub(request).prune(keep_last=keep_last, keep_trusted=keep_trusted, dry_run=dry_run)
