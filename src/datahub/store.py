# -*- coding: utf-8 -*-
"""数据中台的持久化层：SQLite（标准库 sqlite3，零新依赖）。

为什么是 SQLite 而不是 PostgreSQL / MySQL：
    这是单机作品集项目，装一个数据库只为了存几张元数据表，运维成本远大于收益；
    但「元数据必须落盘」这件事不能省——评测要可追溯，就不能只放内存 dict。
    SQLite 在这里的定位是**嵌入式元数据存储**：单文件、随仓库走、不需要额外进程。

为什么每次操作都新建连接：
    FastAPI 的同步路由跑在线程池里，一个跨线程共享的 sqlite3.Connection 默认禁止使用
    （check_same_thread），放开又得自己管并发。SQLite 打开连接的成本在毫秒级，
    本项目写入频率极低（同步数据、跑评测时各写一次），所以「一次操作一个连接」最简单也最安全。
    真要上多实例，把这一层换成 SQLAlchemy + PG 即可，上层不感知。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .models import (
    DatasetKind,
    DatasetSpec,
    DatasetVersion,
    LineageEdge,
    LineageGraph,
    QualityIssue,
    QualityReport,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL,
    title       TEXT DEFAULT '',
    owner       TEXT DEFAULT '',
    tags        TEXT DEFAULT '[]',
    description TEXT DEFAULT '',
    depends_on  TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_id     TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    rows           INTEGER DEFAULT 0,
    size_bytes     INTEGER DEFAULT 0,
    quality_status TEXT DEFAULT 'ok',
    registered_at  TEXT NOT NULL,
    PRIMARY KEY (dataset_id, content_hash)
);

CREATE TABLE IF NOT EXISTS quality_reports (
    dataset_id   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    checked_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    errors       TEXT DEFAULT '[]',
    warnings     TEXT DEFAULT '[]',
    stats        TEXT DEFAULT '{}',
    PRIMARY KEY (dataset_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_versions_dataset ON dataset_versions(dataset_id, registered_at);
"""


class DataHubStore:
    """元数据仓储。所有方法都是同步的——调用量小，不值得引入 async。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------- 数据集登记 ---------------- #
    def upsert_spec(self, spec: DatasetSpec) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO datasets (id, kind, path, title, owner, tags, description, depends_on)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     kind=excluded.kind, path=excluded.path, title=excluded.title,
                     owner=excluded.owner, tags=excluded.tags,
                     description=excluded.description, depends_on=excluded.depends_on""",
                (
                    spec.id,
                    spec.kind.value,
                    spec.path,
                    spec.title,
                    spec.owner,
                    json.dumps(spec.tags, ensure_ascii=False),
                    spec.description,
                    json.dumps(spec.depends_on, ensure_ascii=False),
                ),
            )

    def get_spec(self, dataset_id: str) -> Optional[DatasetSpec]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return _row_to_spec(row) if row else None

    def list_specs(self) -> List[DatasetSpec]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY id").fetchall()
        return [_row_to_spec(r) for r in rows]

    # ---------------- 版本 ---------------- #
    def add_version(self, version: DatasetVersion) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO dataset_versions
                   (dataset_id, content_hash, rows, size_bytes, quality_status, registered_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(dataset_id, content_hash) DO UPDATE SET
                     rows=excluded.rows, size_bytes=excluded.size_bytes,
                     quality_status=excluded.quality_status""",
                (
                    version.dataset_id,
                    version.content_hash,
                    version.rows,
                    version.size_bytes,
                    version.quality_status,
                    version.registered_at,
                ),
            )

    def get_version(self, dataset_id: str, content_hash: str) -> Optional[DatasetVersion]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM dataset_versions WHERE dataset_id=? AND content_hash=?",
                (dataset_id, content_hash),
            ).fetchone()
        return _row_to_version(row) if row else None

    def latest_version(self, dataset_id: str) -> Optional[DatasetVersion]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM dataset_versions WHERE dataset_id=?
                   ORDER BY registered_at DESC, rowid DESC LIMIT 1""",
                (dataset_id,),
            ).fetchone()
        return _row_to_version(row) if row else None

    def list_versions(self, dataset_id: str) -> List[DatasetVersion]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_versions WHERE dataset_id=? ORDER BY registered_at DESC",
                (dataset_id,),
            ).fetchall()
        return [_row_to_version(r) for r in rows]

    def count_versions(self, dataset_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM dataset_versions WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
        return int(row["c"]) if row else 0

    # ---------------- 质量报告 ---------------- #
    def save_quality(self, report: QualityReport) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO quality_reports
                   (dataset_id, content_hash, checked_at, status, errors, warnings, stats)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(dataset_id, content_hash) DO UPDATE SET
                     checked_at=excluded.checked_at, status=excluded.status,
                     errors=excluded.errors, warnings=excluded.warnings, stats=excluded.stats""",
                (
                    report.dataset_id,
                    report.content_hash,
                    report.checked_at,
                    report.status,
                    json.dumps([e.model_dump() for e in report.errors], ensure_ascii=False),
                    json.dumps([w.model_dump() for w in report.warnings], ensure_ascii=False),
                    json.dumps(report.stats, ensure_ascii=False),
                ),
            )

    def get_quality(self, dataset_id: str, content_hash: str) -> Optional[QualityReport]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM quality_reports WHERE dataset_id=? AND content_hash=?",
                (dataset_id, content_hash),
            ).fetchone()
        if not row:
            return None
        return QualityReport(
            dataset_id=row["dataset_id"],
            content_hash=row["content_hash"],
            checked_at=row["checked_at"],
            errors=[QualityIssue(**e) for e in json.loads(row["errors"] or "[]")],
            warnings=[QualityIssue(**w) for w in json.loads(row["warnings"] or "[]")],
            stats=json.loads(row["stats"] or "{}"),
        )

    # ---------------- 血缘 ---------------- #
    def lineage(self) -> LineageGraph:
        specs = self.list_specs()
        edges: List[LineageEdge] = []
        for spec in specs:
            for dep in spec.depends_on:
                edges.append(
                    LineageEdge(from_dataset=spec.id, to_dataset=dep, field="depends_on", count=1)
                )
        return LineageGraph(nodes=[s.id for s in specs], edges=edges)

    def referenced_by(self, dataset_id: str) -> List[str]:
        return [e.from_dataset for e in self.lineage().edges if e.to_dataset == dataset_id]


# --------------------------------------------------------------------------- #
# 行 -> 模型
# --------------------------------------------------------------------------- #
def _row_to_spec(row: sqlite3.Row) -> DatasetSpec:
    return DatasetSpec(
        id=row["id"],
        kind=DatasetKind(row["kind"]),
        path=row["path"],
        title=row["title"] or "",
        owner=row["owner"] or "",
        tags=json.loads(row["tags"] or "[]"),
        description=row["description"] or "",
        depends_on=json.loads(row["depends_on"] or "[]"),
    )


def _row_to_version(row: sqlite3.Row) -> DatasetVersion:
    return DatasetVersion(
        dataset_id=row["dataset_id"],
        content_hash=row["content_hash"],
        rows=row["rows"],
        size_bytes=row["size_bytes"],
        quality_status=row["quality_status"],
        registered_at=row["registered_at"],
    )


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}
