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


class EvalHubStore:
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

    def save_run(self, record: EvalRunRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO eval_runs
                   (id, task, dataset_id, dataset_hash, status, verdict, trusted,
                    config_fingerprint, fingerprint_detail, metrics, warnings, error,
                    started_at, finished_at, duration_ms, cases, gate)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status, verdict=excluded.verdict,
                     metrics=excluded.metrics, warnings=excluded.warnings,
                     error=excluded.error, finished_at=excluded.finished_at,
                     duration_ms=excluded.duration_ms, cases=excluded.cases,
                     gate=excluded.gate""",
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


def _row_to_record(row: sqlite3.Row) -> EvalRunRecord:
    gate_raw = row["gate"] or ""
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
        metrics=json.loads(row["metrics"] or "{}"),
        warnings=json.loads(row["warnings"] or "[]"),
        error=row["error"] or "",
        started_at=row["started_at"],
        finished_at=row["finished_at"] or "",
        duration_ms=row["duration_ms"] or 0.0,
        cases=row["cases"] or 0,
        gate=GateResult.model_validate_json(gate_raw) if gate_raw else None,
    )
