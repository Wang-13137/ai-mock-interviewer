# -*- coding: utf-8 -*-
"""数据中台：把散落的 JSON 数据变成可登记、可校验、可追溯的数据资产。

模块结构：
    models.py    数据契约（Dataset / Version / QualityReport / Lineage）
    store.py     SQLite 元数据仓储
    quality.py   按 kind 的质量门禁（含跨数据集引用完整性）
    registry.py  编排：同步、版本、快照、血缘
    api.py       REST 路由（挂在 /v1/data/*）

最短用法：
    from src.datahub import DataHub
    hub = DataHub(load_settings())
    hub.sync_all()
    print(hub.list_datasets())
"""

from .models import (
    DatasetKind,
    DatasetRecord,
    DatasetSpec,
    DatasetVersion,
    LineageEdge,
    LineageGraph,
    QualityIssue,
    QualityReport,
    SyncResult,
)
from .registry import DataHub, build_default_specs, content_hash
from .store import DataHubStore

__all__ = [
    "DataHub",
    "DataHubStore",
    "DatasetKind",
    "DatasetRecord",
    "DatasetSpec",
    "DatasetVersion",
    "LineageEdge",
    "LineageGraph",
    "QualityIssue",
    "QualityReport",
    "SyncResult",
    "build_default_specs",
    "content_hash",
]
