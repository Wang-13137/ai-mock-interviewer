# -*- coding: utf-8 -*-
"""评测中台 REST 路由（/v1/eval/*）。

一个刻意的设计：`POST /v1/eval/runs` 是**同步执行**并返回完整结果，
而不是丢后台队列 + 轮询。理由：本项目的评测规模是 60 条 query / 10 条对话，
秒级到分钟级；引入任务队列只会增加没人用的复杂度。
真要做长任务，把 runner 换成后台任务即可，路由契约不用变。
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from .models import CaseRow, EvalRunRecord, EvalRunRequest, GateResult, TaskInfo

router = APIRouter(prefix="/v1/eval", tags=["eval"])


def _hub(request: Request):
    hub = getattr(request.app.state, "evalhub", None)
    if hub is None:
        raise HTTPException(status_code=503, detail="评测中台未启用（见 config 的 evalhub.enabled）")
    return hub


@router.get("/tasks", response_model=List[TaskInfo], summary="可用评测任务")
def list_tasks(request: Request) -> List[TaskInfo]:
    return _hub(request).tasks()


@router.post("/runs", response_model=EvalRunRecord, summary="执行一次评测")
async def create_run(req: EvalRunRequest, request: Request) -> EvalRunRecord:
    record = await _hub(request).run(req)
    if record.status.value == "rejected":
        # 拒绝运行是业务结论而非服务故障，用 400 让调用方明确知道"没有产出数字"
        raise HTTPException(status_code=400, detail={"run_id": record.id, "reason": record.error})
    return record


@router.get("/runs", response_model=List[EvalRunRecord], summary="历史 run 列表")
def list_runs(
    request: Request,
    task: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=200),
    trusted_only: bool = False,
) -> List[EvalRunRecord]:
    return _hub(request).list_runs(task=task, limit=limit, only_trusted=trusted_only)


@router.get("/runs/{run_id}", response_model=EvalRunRecord, summary="run 详情")
def get_run(run_id: str, request: Request, with_cases: bool = False) -> EvalRunRecord:
    record = _hub(request).get_run(run_id, with_cases=with_cases)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run 不存在：{run_id}")
    return record


@router.get("/runs/{run_id}/cases", response_model=List[CaseRow], summary="逐条明细（归因用）")
def get_cases(run_id: str, request: Request, limit: Optional[int] = None) -> List[CaseRow]:
    hub = _hub(request)
    if hub.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"run 不存在：{run_id}")
    return hub.get_cases(run_id, limit=limit)


@router.post("/runs/{run_id}/gate", response_model=GateResult, summary="与基线比较（回归门禁）")
def gate_run(
    run_id: str,
    request: Request,
    baseline: Optional[str] = Query(default="latest", description="基线 run id，或 latest"),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
) -> GateResult:
    gate = _hub(request).gate_run(run_id, baseline=baseline, threshold=threshold)
    if gate is None:
        raise HTTPException(status_code=404, detail=f"run 不存在：{run_id}")
    return gate
