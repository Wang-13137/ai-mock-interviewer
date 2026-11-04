# -*- coding: utf-8 -*-
"""评测中台的持久化层（SQLite）。

为什么评测结果必须落库，而不是只写 reports/*.md：
    md 报告是给人看的结论，无法被程序比较。回归门禁要回答的是
    「这次改动让 MRR 掉了几个点」——这需要对**结构化的历史指标**做差集。
    只留 md 的话，每次比较都只能靠人肉翻文件，也就不会真的有人去比。

为什么保存逐条明细（eval_cases）：
    指标退步 0.02 只是症状，真正要改代码得知道是哪些 query 掉的。
    没有明细的评测只能证明"变差了"，不能定位"哪里差"，等于没有归因能力。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .models import (
    CaseRow,
    EvalRunRecord,
    GateResult,
    RunStatus,
    Verdict,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id                  TEXT PRIMARY KEY,
    task                TEXT NOT NULL,
    dataset_id          TEXT DEFAULT '',
    dataset_hash        TEXT DEFAULT '',
    status              TEXT NOT NULL,
    verdict             TEXT NOT NULL,
    trusted             INTEGER DEFAULT 0,
    config_fingerprint  TEXT DEFAULT '',
    fingerprint_detail  TEXT DEFAULT '{}',
    metrics             TEXT DEFAULT '{}',
    warnings            TEXT DEFAULT '[]',
    error               TEXT DEFAULT '',
    started_at          TEXT NOT NULL,
    finished_at         TEXT DEFAULT '',
    duration_ms         REAL DEFAULT 0,
    cases               INTEGER DEFAULT 0,
    gate                TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS eval_cases (
    run_id   TEXT NOT NULL,
    case_id  TEXT NOT NULL,
    metrics  TEXT DEFAULT '{}',
    detail   TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_task ON eval_runs(task, started_at);
CREATE INDEX IF NOT EXISTS idx_cases_run ON eval_cases(run_id);
"""

# v0.8 新增列。
# 为什么需要显式迁移：`CREATE TABLE IF NOT EXISTS` 对**已存在的表不会加列**，
# 老库升级后 INSERT 就会因为列不存在报错。真实项目的数据表只会越来越宽，
# 没有迁移机制的中台，第一次加字段就是一次停服。
MIGRATIONS: list[tuple[str, str]] = [
    ("code_version", "ALTER TABLE eval_runs ADD COLUMN code_version TEXT DEFAULT ''"),
    ("runtime", "ALTER TABLE eval_runs ADD COLUMN runtime TEXT DEFAULT '{}'"),
    ("stage", "ALTER TABLE eval_runs ADD COLUMN stage TEXT DEFAULT ''"),
    ("progress", "ALTER TABLE eval_runs ADD COLUMN progress REAL DEFAULT 0"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(eval_runs)").fetchall()}
    for column, ddl in MIGRATIONS:
        if column not in cols:
            conn.execute(ddl)


class EvalHubStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            _migrate(conn)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_run(self, record: EvalRunRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO eval_runs
                   (id, task, dataset_id, dataset_hash, status, verdict, trusted,
                    config_fingerprint, fingerprint_detail, metrics, warnings, error,
                    started_at, finished_at, duration_ms, cases, gate,
                    code_version, runtime, stage, progress)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     dataset_id=excluded.dataset_id,
                     dataset_hash=excluded.dataset_hash,
                     status=excluded.status, verdict=excluded.verdict,
                     trusted=excluded.trusted,
                     config_fingerprint=excluded.config_fingerprint,
                     fingerprint_detail=excluded.fingerprint_detail,
                     metrics=excluded.metrics, warnings=excluded.warnings,
                     error=excluded.error, finished_at=excluded.finished_at,
                     duration_ms=excluded.duration_ms, cases=excluded.cases,
                     gate=excluded.gate, stage=excluded.stage, progress=excluded.progress""",
                (
                    record.id,
                    record.task,
                    record.dataset_id,
                    record.dataset_hash,
                    record.status.value,
                    record.verdict.value,
                    1 if record.trusted else 0,
                    record.config_fingerprint,
                    json.dumps(record.fingerprint_detail, ensure_ascii=False),
                    json.dumps(record.metrics, ensure_ascii=False),
                    json.dumps(record.warnings, ensure_ascii=False),
                    record.error,
                    record.started_at,
                    record.finished_at,
                    record.duration_ms,
                    record.cases,
                    record.gate.model_dump_json() if record.gate else "",
                    record.code_version,
                    json.dumps(record.runtime, ensure_ascii=False),
                    record.stage,
                    record.progress,
                ),
            )

    def save_cases(self, run_id: str, rows: List[CaseRow]) -> None:
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO eval_cases (run_id, case_id, metrics, detail) VALUES (?,?,?,?)",
                [
                    (
                        run_id,
                        r.case_id,
                        json.dumps(r.metrics, ensure_ascii=False),
                        json.dumps(r.detail, ensure_ascii=False),
                    )
                    for r in rows
                ],
            )

    def get_run(self, run_id: str, with_cases: bool = False) -> Optional[EvalRunRecord]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                return None
            record = _row_to_record(row)
            if with_cases:
                rows = conn.execute(
                    "SELECT * FROM eval_cases WHERE run_id=? ORDER BY rowid", (run_id,)
                ).fetchall()
                record.cases = len(rows)
        return record

    def list_cases(self, run_id: str, limit: Optional[int] = None) -> List[CaseRow]:
        sql = "SELECT * FROM eval_cases WHERE run_id=? ORDER BY rowid"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._conn() as conn:
            rows = conn.execute(sql, (run_id,)).fetchall()
        return [
            CaseRow(
                case_id=r["case_id"],
                metrics=json.loads(r["metrics"] or "{}"),
                detail=json.loads(r["detail"] or "{}"),
            )
            for r in rows
        ]

    def list_runs(
        self, task: Optional[str] = None, limit: int = 20, only_trusted: bool = False
    ) -> List[EvalRunRecord]:
        sql = "SELECT * FROM eval_runs"
        clauses: List[str] = []
        params: List[Any] = []
        if task:
            clauses.append("task=?")
            params.append(task)
        if only_trusted:
            clauses.append("trusted=1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
        params.append(int(limit))
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def set_verdict(self, run_id: str, gate: GateResult) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE eval_runs SET verdict=?, gate=? WHERE id=?",
                (gate.verdict.value, gate.model_dump_json(), run_id),
            )

    # ---------------- 保留策略 ---------------- #
    def prune(self, keep_last: int = 50, keep_trusted: int = 20, dry_run: bool = False) -> Dict[str, Any]:
        """按任务裁剪历史：保留最近 `keep_last` 条 + 最近 `keep_trusted` 个 trusted 基线。

        为什么评测库需要保留策略：每次 run 都落库 + 落逐条明细，
        60 条/次的规模下跑几百次就是几万行。列表查询与 CI 都会慢慢变慢，
        而真正有价值的只有「最近的状态」和「可信基线」。**在跑的 run 永不删除。**
        """
        with self._conn() as conn:
            tasks = [r["task"] for r in conn.execute("SELECT DISTINCT task FROM eval_runs").fetchall()]
            keep: set[str] = set()
            for task in tasks:
                rows = conn.execute(
                    "SELECT id, trusted FROM eval_runs WHERE task=? ORDER BY started_at DESC, rowid DESC",
                    (task,),
                ).fetchall()
                keep.update(r["id"] for r in rows[: max(0, keep_last)])
                keep.update([r["id"] for r in rows if r["trusted"]][: max(0, keep_trusted)])
            all_rows = conn.execute("SELECT id, status FROM eval_runs").fetchall()
            doomed = [
                r["id"]
                for r in all_rows
                if r["id"] not in keep and r["status"] not in ("pending", "running")
            ]
            if doomed and not dry_run:
                marks = ",".join("?" * len(doomed))
                conn.execute(f"DELETE FROM eval_cases WHERE run_id IN ({marks})", doomed)
                conn.execute(f"DELETE FROM eval_runs WHERE id IN ({marks})", doomed)
        return {"deleted": len(doomed), "kept": len(keep), "dry_run": dry_run}

    def mark_stale_runs(self, reason: str = "") -> List[str]:
        """把所有非终态 run 落为 failed，返回被清理的 run id 列表。

        为什么需要这一步：后台任务活在进程内存里，进程被 kill / 崩溃后，
        `pending`/`running` 的 run 永远等不到执行者——它们不是「还在跑」，
        而是永远不会跑。不清理的后果是 `baseline=latest` 之类的逻辑可能
        被一条僵尸 run 卡住，而且列表页会一直显示假象。启动时清扫一次，
        比「假装它们还在跑」诚实。
        """
        err = reason or "进程重启：这个 run 在进程退出前未完成，无从恢复，标记为失败（没有产出任何数字）"
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM eval_runs WHERE status IN ('pending', 'running')"
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                conn.execute(
                    """UPDATE eval_runs
                       SET status='failed', error=?, stage='', finished_at=?
                       WHERE status IN ('pending', 'running')""",
                    (err, now),
                )
        return ids


def _row_to_record(row: sqlite3.Row) -> EvalRunRecord:
    gate_raw = row["gate"] or ""
    keys = row.keys()
    return EvalRunRecord(
        id=row["id"],
        task=row["task"],
        dataset_id=row["dataset_id"] or "",
        dataset_hash=row["dataset_hash"] or "",
        status=RunStatus(row["status"]),
        verdict=Verdict(row["verdict"]),
        trusted=bool(row["trusted"]),
        config_fingerprint=row["config_fingerprint"] or "",
        fingerprint_detail=json.loads(row["fingerprint_detail"] or "{}"),
        code_version=(row["code_version"] or "") if "code_version" in keys else "",
        runtime=json.loads((row["runtime"] or "{}") if "runtime" in keys else "{}"),
        metrics=json.loads(row["metrics"] or "{}"),
        warnings=json.loads(row["warnings"] or "[]"),
        error=row["error"] or "",
        started_at=row["started_at"],
        finished_at=row["finished_at"] or "",
        duration_ms=row["duration_ms"] or 0.0,
        cases=row["cases"] or 0,
        stage=(row["stage"] or "") if "stage" in keys else "",
        progress=(row["progress"] or 0.0) if "progress" in keys else 0.0,
        gate=GateResult.model_validate_json(gate_raw) if gate_raw else None,
    )
