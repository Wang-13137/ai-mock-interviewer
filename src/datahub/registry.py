# -*- coding: utf-8 -*-
"""数据中台的编排层。

职责边界：
    · 这里只做「元数据」——数据本身仍然以 JSON 文件为准，不做数据搬迁。
      作品集项目里去搞一套 HDFS/数据湖是过度设计；中台要解决的是**可追溯与可控**，
      不是把 22KB 的题库搬进数据库。
    · 质量门禁在**写入侧**（登记时）而不是消费侧（评测时）。
      坏数据进不来，比每次评测都过滤一遍便宜，也避免"忘了过滤"。

三个关键语义：
    1. content_hash 幂等：同一份内容重复同步 = unchanged，不会产生新版本，也不重复跑校验。
    2. blocked 不覆盖 latest：数据有 error 时不许顶掉上一个可用版本，避免"改坏了还生效"。
    3. dirty 探测：磁盘 hash ≠ 已登记最新 hash 时标 dirty——提醒"文件改了但没同步"，
       这正是评测结果对不上号的常见原因。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core.config import Settings
from .models import (
    DatasetKind,
    DatasetRecord,
    DatasetSpec,
    DatasetVersion,
    LineageGraph,
    QualityReport,
    SyncResult,
)
from .quality import check, extract_ids
from .store import DataHubStore

log = logging.getLogger("aimi.datahub")

REPO_ROOT = Path(__file__).resolve().parents[2]


def content_hash(raw: bytes) -> str:
    """内容指纹：sha256 前 16 位。

    为什么不用语义化版本号（v1/v2）：人会忘记手动递增，而且两个分支可能都叫 v2。
    hash 是**唯一能证明「评测跑的是这堆字节」**的东西，也不依赖人守规矩。

    为什么先归一化换行符再算：同一份 JSON 在 Windows 是 CRLF、在 Linux CI 是
    LF，字节不同但语义完全相同——不归一的话，跨平台环境永远判 dirty /
    incomparable，门禁会在"数据没变"时集体误报（2026-11-05 CI 实测踩中）。
    """
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()[:16]


def build_default_specs(settings: Settings) -> List[DatasetSpec]:
    """项目内置的 4 份数据资产。路径跟随配置，避免硬编码。"""
    return [
        DatasetSpec(
            id="question_bank",
            kind=DatasetKind.QUESTION_BANK,
            path=settings.agent.question_bank,
            title="面试题库（检索语料源）",
            tags=["corpus", "rag"],
            description="被 search_question_bank 工具与检索评测同时消费；改 id 会影响评测集引用完整性。",
        ),
        DatasetSpec(
            id="retrieval_eval",
            kind=DatasetKind.RETRIEVAL_EVAL,
            path="config/retrieval_eval.json",
            title="检索评测集（60 条 query → 相关题号）",
            tags=["eval", "retrieval"],
            description="人工标注，不是从 tags 自动生成——自动生成等于答案泄露。",
            depends_on=["question_bank"],
        ),
        DatasetSpec(
            id="eval_set",
            kind=DatasetKind.EVAL_SET,
            path="config/eval_set.example.json",
            title="对话评测集（LLM-as-judge 五维）",
            tags=["eval", "generation"],
            description="生成侧评测用例；human_score 用于算 kappa/QWK 一致性。",
        ),
        DatasetSpec(
            id="bench_questions",
            kind=DatasetKind.BENCH,
            path="config/bench_questions.json",
            title="时延基准对话脚本",
            tags=["bench", "latency"],
            description="bench 用的固定问句，保证各次时延测量同输入。",
        ),
    ]


class DataHub:
    """数据资产管理器：登记、校验、版本、血缘。"""

    def __init__(
        self,
        settings: Settings,
        specs: Optional[Sequence[DatasetSpec]] = None,
        db_path: Optional[str | Path] = None,
        root: Optional[str | Path] = None,
    ) -> None:
        self.settings = settings
        self.root = Path(root) if root else REPO_ROOT
        self.store = DataHubStore(db_path or settings.datahub.db_path)
        self.specs: Dict[str, DatasetSpec] = {}
        for spec in specs if specs is not None else build_default_specs(settings):
            self.specs[spec.id] = spec
        for spec in self.specs.values():
            self.store.upsert_spec(spec)

    # ---------------- 路径与读取 ---------------- #
    def _abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def _read_json(self, rel: str) -> Any:
        return json.loads(self._abs(rel).read_text(encoding="utf-8"))

    # 评测中台通过这两个方法取数据，避免它自己拼路径（路径权威只在中台一处）
    def path_of(self, dataset_id: str) -> Path:
        spec = self.specs.get(dataset_id) or self.store.get_spec(dataset_id)
        if spec is None:
            raise KeyError(f"未登记的 dataset: {dataset_id}")
        return self._abs(spec.path)

    def load_json(self, dataset_id: str) -> Any:
        return self._read_json(str(self.path_of(dataset_id)))

    # ---------------- 同步 ---------------- #
    def sync(self, dataset_id: str, force: bool = False) -> SyncResult:
        """同步一份数据集：算 hash -> 校验 -> 落库。幂等。"""
        spec = self.specs.get(dataset_id) or self.store.get_spec(dataset_id)
        if spec is None:
            return SyncResult(dataset_id=dataset_id, status="missing", message="未登记的数据集")

        path = self._abs(spec.path)
        if not path.exists():
            return SyncResult(dataset_id=dataset_id, status="missing", message=f"文件不存在：{spec.path}")

        raw = path.read_bytes()
        digest = content_hash(raw)

        # 幂等：内容没变就什么都不做（不重复校验、不产生新版本）
        existing = self.store.get_version(dataset_id, digest)
        if existing is not None and not force:
            return SyncResult(
                dataset_id=dataset_id,
                content_hash=digest,
                status="unchanged",
                rows=existing.rows,
                quality=self.store.get_quality(dataset_id, digest),
                message="内容未变",
            )

        data = json.loads(raw.decode("utf-8"))

        # 引用完整性：先把依赖数据集的合法主键取出来
        ctx: Dict[str, Any] = {}
        ref_ids = self._reference_ids(spec)
        if ref_ids is not None:
            ctx["ref_ids"] = ref_ids

        report = check(spec.id, spec.kind, digest, data, ctx)
        self.store.save_quality(report)

        if not report.passed:
            # 关键语义：blocked 不写版本，因此不会顶掉上一个可用版本
            return SyncResult(
                dataset_id=dataset_id,
                content_hash=digest,
                status="blocked",
                rows=report.stats.get("rows", 0),
                quality=report,
                message=f"质量门禁拦截：{len(report.errors)} 个 error",
            )

        version = DatasetVersion(
            dataset_id=dataset_id,
            content_hash=digest,
            rows=int(report.stats.get("rows", 0) or 0),
            size_bytes=len(raw),
            quality_status=report.status,
        )
        self.store.add_version(version)
        log.info("datahub 登记 %s hash=%s rows=%s status=%s", dataset_id, digest, version.rows, report.status)
        return SyncResult(
            dataset_id=dataset_id,
            content_hash=digest,
            status="registered",
            rows=version.rows,
            quality=report,
            message=report.summary(),
        )

    def _reference_ids(self, spec: DatasetSpec) -> Optional[set]:
        """取依赖数据集的主键集合；任一层缺失就返回 None（表示无法做完整性检查）。"""
        if not spec.depends_on:
            return None
        ids: set = set()
        for dep_id in spec.depends_on:
            dep = self.specs.get(dep_id) or self.store.get_spec(dep_id)
            if dep is None:
                return None
            try:
                data = self._read_json(dep.path)
            except (OSError, json.JSONDecodeError):
                return None
            ids.update(extract_ids(dep.kind, data))
        return ids or None

    def sync_all(self, force: bool = False) -> List[SyncResult]:
        """按依赖拓扑顺序同步：被依赖的先登记，否则引用完整性无从校验。"""
        results: List[SyncResult] = []
        for dataset_id in self._topo_order():
            results.append(self.sync(dataset_id, force=force))
        return results

    def _topo_order(self) -> List[str]:
        order: List[str] = []
        seen: set = set()
        visiting: set = set()

        def visit(node: str) -> None:
            if node in seen or node in visiting:
                return
            visiting.add(node)
            spec = self.specs.get(node)
            if spec:
                for dep in spec.depends_on:
                    visit(dep)
            visiting.discard(node)
            seen.add(node)
            order.append(node)

        for node in self.specs:
            visit(node)
        return order

    # ---------------- 查询 ---------------- #
    def list_datasets(self) -> List[DatasetRecord]:
        return [self._record(spec.id) for spec in self._ordered_specs()]

    def get(self, dataset_id: str) -> Optional[DatasetRecord]:
        if dataset_id not in self.specs and self.store.get_spec(dataset_id) is None:
            return None
        return self._record(dataset_id)

    def _ordered_specs(self) -> List[DatasetSpec]:
        stored = {s.id: s for s in self.store.list_specs()}
        merged = {**stored, **self.specs}
        return [merged[k] for k in sorted(merged)]

    def _record(self, dataset_id: str) -> DatasetRecord:
        spec = self.specs.get(dataset_id) or self.store.get_spec(dataset_id)
        assert spec is not None
        latest = self.store.latest_version(dataset_id)
        record = DatasetRecord(
            spec=spec,
            latest=latest,
            versions=self.store.count_versions(dataset_id),
            referenced_by=self.store.referenced_by(dataset_id),
            missing=not self._abs(spec.path).exists(),
        )
        if latest is not None:
            record.quality = self.store.get_quality(dataset_id, latest.content_hash)
            # dirty：磁盘内容与已登记的最新版本不一致（改了但没同步）
            if not record.missing:
                on_disk = content_hash(self._abs(spec.path).read_bytes())
                record.dirty = on_disk != latest.content_hash
        return record

    def lineage(self) -> LineageGraph:
        return self.store.lineage()

    def snapshot(self, dataset_id: str) -> Dict[str, Any]:
        """给评测中台用的快照：绑定 hash 而非路径。

        返回的 `trusted_data` 表示「已登记且质量过关」。评测若引用未登记/拦截的数据，
        必须显式记录，否则事后无法解释那批数字是从哪来的。
        """
        record = self.get(dataset_id)
        if record is None:
            return {"dataset_id": dataset_id, "registered": False, "trusted_data": False,
                    "content_hash": "", "rows": 0, "quality_status": "unknown"}
        latest = record.latest
        return {
            "dataset_id": dataset_id,
            "registered": latest is not None,
            "trusted_data": latest is not None and latest.quality_status in ("ok", "warning") and not record.dirty,
            "content_hash": latest.content_hash if latest else "",
            "rows": latest.rows if latest else 0,
            "quality_status": latest.quality_status if latest else "unknown",
            "dirty": record.dirty,
        }
