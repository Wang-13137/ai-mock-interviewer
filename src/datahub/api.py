# -*- coding: utf-8 -*-
"""数据中台 REST 路由（/v1/data/*）。

依赖注入用 `request.app.state.datahub`：中台实例在 lifespan 里构造一次，
路由不持有全局单例，测试可以直接替换 app.state 而不用打补丁。

未启用（datahub.enabled=false）或初始化失败时一律返回 503 而不是 500——
面试官 clone 下来跑，即使数据库目录只读也不该让服务起不来。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from .models import DatasetRecord, LineageGraph, QualityReport, SyncResult

router = APIRouter(prefix="/v1/data", tags=["data"])


def _hub(request: Request):
    hub = getattr(request.app.state, "datahub", None)
    if hub is None:
        raise HTTPException(status_code=503, detail="数据中台未启用（见 config 的 datahub.enabled）")
    return hub


@router.get("/datasets", response_model=List[DatasetRecord], summary="列出已登记的数据集")
def list_datasets(request: Request) -> List[DatasetRecord]:
    return _hub(request).list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetRecord, summary="数据集详情")
def get_dataset(dataset_id: str, request: Request) -> DatasetRecord:
    record = _hub(request).get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"未登记的 dataset: {dataset_id}")
    return record


@router.get("/datasets/{dataset_id}/quality", response_model=QualityReport, summary="最新版本的质量报告")
def get_quality(dataset_id: str, request: Request) -> QualityReport:
    record = _hub(request).get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"未登记的 dataset: {dataset_id}")
    if record.quality is None:
        raise HTTPException(status_code=404, detail=f"{dataset_id} 尚无质量报告，先调用 sync")
    return record.quality


@router.post("/datasets/{dataset_id}/sync", response_model=SyncResult, summary="同步单个数据集")
def sync_dataset(
    dataset_id: str,
    request: Request,
    force: bool = Query(default=False, description="强制重新校验（忽略 hash 幂等）"),
) -> SyncResult:
    hub = _hub(request)
    result = hub.sync(dataset_id, force=force)
    if result.status == "missing" and hub.get(dataset_id) is None:
        raise HTTPException(status_code=404, detail=result.message)
    return result


@router.post("/sync", response_model=List[SyncResult], summary="按依赖顺序全量同步")
def sync_all(request: Request, force: bool = Query(default=False)) -> List[SyncResult]:
    return _hub(request).sync_all(force=force)


@router.get("/lineage", response_model=LineageGraph, summary="数据集血缘（谁引用谁）")
def lineage(request: Request) -> LineageGraph:
    return _hub(request).lineage()


@router.get("/snapshot/{dataset_id}", summary="评测可用的数据快照（绑定 content_hash）")
def snapshot(dataset_id: str, request: Request) -> Dict[str, Any]:
    return _hub(request).snapshot(dataset_id)
