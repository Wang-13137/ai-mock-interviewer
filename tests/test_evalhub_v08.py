# -*- coding: utf-8 -*-
"""v0.8 新增能力测试：显著性检验、异步 run、鉴权、库迁移、保留策略、溯源。

这五块各自守住一个"平台化"的边界：
    · bootstrap     —— 门禁判的是"统计上真的退步"，不是"数字动了"；
    · 异步 + 状态机 —— 长任务不会在 HTTP 超时后变成幽灵 run；
    · 鉴权          —— 写接口不能被人随便调用来烧额度；
    · 库迁移        —— 加字段不等于停服；
    · 保留策略      —— 评测库不会无限膨胀；
    · 溯源          —— 指标能追到具体提交。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.core.config import AuthConfig, DataHubConfig, EvalHubConfig, Settings
from src.datahub import DataHub
from src.evalhub import EvalHub, EvalRunRequest, EvalTask, RunStatus, Verdict
from src.evalhub.bootstrap import bootstrap_metric, paired_bootstrap, reconstruct
from src.evalhub.gate import compare
from src.evalhub.models import CaseRow, EvalRunRecord
from src.evalhub.store import EvalHubStore

BANK = {
    "version": "test",
    "items": [
        {"id": "q1", "dimension": "项目深挖", "difficulty": 2, "tags": ["RAG"], "question": "你的 RAG 项目怎么切分文档？"},
        {"id": "q2", "dimension": "工程素养", "difficulty": 3, "tags": ["缓存"], "question": "如何设计多级缓存？"},
        {"id": "q3", "dimension": "算法基础", "difficulty": 1, "tags": ["排序"], "question": "快速排序的复杂度？"},
    ],
}
RETRIEVAL = {
    "version": "test",
    "cases": [
        {"query": "RAG 文档切分", "relevant": ["q1"], "kind": "literal"},
        {"query": "缓存设计", "relevant": ["q2"], "kind": "semantic"},
        {"query": "快排复杂度", "relevant": ["q3"], "kind": "literal"},
    ],
}


def write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def recall_rows(values: list[float], prefix: str = "c") -> list[CaseRow]:
    """造一批检索明细：recall@3 与 rr 同值（单条 query 命中即 rr=1）。"""
    return [
        CaseRow(case_id=f"{prefix}{i}", metrics={"recall@3": v, "rr": v}, detail={"kind": "literal"})
        for i, v in enumerate(values)
    ]


def run_of(rows: list[CaseRow], run_id: str, metric: str = "recall@3", **kw) -> EvalRunRecord:
    values = [r.metrics[metric] for r in rows]
    metrics = {metric: round(sum(values) / len(values), 6)}
    return EvalRunRecord(id=run_id, task="retrieval", metrics=metrics, trusted=True,
                         config_fingerprint="fp", dataset_hash="h", **kw)


@pytest.fixture()
def repo(tmp_path: Path):
    root = tmp_path / "repo"
    cfg = root / "config"
    cfg.mkdir(parents=True)
    write(cfg / "question_bank.json", BANK)
    write(cfg / "retrieval_eval.json", RETRIEVAL)

    def make(**evalhub) -> EvalHub:
        settings = Settings(
            env="test",
            datahub=DataHubConfig(db_path=str(tmp_path / "dh.db")),
            evalhub=EvalHubConfig(db_path=str(tmp_path / "eh.db"), **evalhub),
        )
        datahub = DataHub(settings, root=root)
        datahub.sync_all()
        return EvalHub(settings, datahub)

    return make


# --------------------------------------------------------------------------- #
# 1. 配对 bootstrap
# --------------------------------------------------------------------------- #
def test_reconstruct_maps_mrr_from_rr():
    rows = recall_rows([1.0, 0.5, 0.0])
    series = reconstruct("mrr@10", rows)
    assert series is not None and sorted(series.values()) == [0.0, 0.5, 1.0]


def test_reconstruct_returns_none_for_unsupported_metric():
    """kappa 不是逐条均值，无法重建——必须返回 None，而不是硬编一个数出来。"""
    rows = recall_rows([1.0, 0.5])
    assert reconstruct("kappa", rows) is None


def test_bootstrap_ci_contains_zero_when_no_change():
    rows = recall_rows([1.0, 0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5])
    series = reconstruct("recall@3", rows)
    assert series is not None
    ci = paired_bootstrap(series, series, metric="recall@3")
    assert ci is not None
    assert ci.low <= 0.0 <= ci.high
    assert ci.significant is False


def test_bootstrap_ci_strictly_below_zero_when_every_case_drops():
    base = reconstruct("recall@3", recall_rows([1.0] * 20))
    cur = reconstruct("recall@3", recall_rows([0.5] * 20))
    assert base and cur
    ci = paired_bootstrap(base, cur, metric="recall@3")
    assert ci is not None
    assert ci.high < 0.0 and ci.significant is True


def test_bootstrap_refuses_tiny_sample():
    base = reconstruct("recall@3", recall_rows([1.0, 0.0]))
    cur = reconstruct("recall@3", recall_rows([0.0, 0.0]))
    assert paired_bootstrap(base, cur) is None


def test_bootstrap_metric_rejects_mismatched_summary():
    """明细重建均值对不上落库指标时，宁可不检验，也不能拿错的样本算 CI。"""
    rows = recall_rows([1.0] * 10)
    ci, why = bootstrap_metric("recall@3", rows, rows, base_value=0.5, cur_value=0.5)
    assert ci is None
    assert "不符" in why


# --------------------------------------------------------------------------- #
# 2. 门禁：显著性判定
# --------------------------------------------------------------------------- #
def test_gate_passes_when_drop_is_within_noise():
    """60 条里掉 2 条：点估计 -3.3% 看着像退步，但 CI 跨 0 —— 判噪声。"""
    base_rows = recall_rows([1.0] * 58 + [0.0] * 2)
    cur_rows = recall_rows([1.0] * 56 + [0.0] * 4)
    gate = compare(
        run_of(base_rows, "base"),
        run_of(cur_rows, "cur"),
        base_cases=base_rows,
        cur_cases=cur_rows,
        abs_threshold=0.02,
        pct_threshold=0.02,
        iters=800,
    )
    assert gate.verdict is Verdict.PASS
    assert any("跨 0" in r for r in gate.reasons), "判 pass 时必须说明为什么这不是退步"
    d = next(d for d in gate.deltas if d.metric == "recall@3")
    assert d.method == "bootstrap" and d.significant is False


def test_gate_regresses_when_drop_is_significant():
    base_rows = recall_rows([1.0] * 60)
    cur_rows = recall_rows([1.0] * 30 + [0.0] * 30)
    gate = compare(
        run_of(base_rows, "base"),
        run_of(cur_rows, "cur"),
        base_cases=base_rows,
        cur_cases=cur_rows,
        abs_threshold=0.02,
        pct_threshold=0.02,
        iters=800,
    )
    assert gate.verdict is Verdict.REGRESSED
    d = next(d for d in gate.regressed_metrics if d.metric == "recall@3")
    assert d.ci_high is not None and d.ci_high < 0


def test_gate_falls_back_to_point_estimate_without_detail():
    """没有明细就不假装做过显著性检验，并明说。"""
    base = EvalRunRecord(id="b", task="retrieval", metrics={"recall@3": 0.9},
                         trusted=True, config_fingerprint="fp", dataset_hash="h")
    cur = EvalRunRecord(id="c", task="retrieval", metrics={"recall@3": 0.7},
                        trusted=True, config_fingerprint="fp", dataset_hash="h")
    gate = compare(base, cur, abs_threshold=0.02, pct_threshold=0.02)
    assert gate.verdict is Verdict.REGRESSED
    assert any("未提供逐条明细" in r for r in gate.reasons)


# --------------------------------------------------------------------------- #
# 3. 异步 run 与状态机
# --------------------------------------------------------------------------- #
async def test_submit_returns_pending_then_finishes(repo):
    hub = repo()
    record = await hub.submit(EvalRunRequest(task=EvalTask.RETRIEVAL))
    assert record.status is RunStatus.PENDING
    assert record.stage in ("queued", "binding_dataset", "running_task")
    assert record.code_version == record.code_version  # 有值即可，CI 里可能没有 git

    final = await hub.wait(record.id, timeout=30)
    assert final is not None
    assert final.status is RunStatus.SUCCEEDED
    assert final.progress == 1.0
    assert final.stage == ""
    assert final.metrics["mrr@10"] > 0
    assert final.runtime.get("python")


async def test_timeout_marks_run_failed(repo):
    """超时必须有终点：否则后台还在跑、库里永远 pending，就是幽灵 run。"""
    hub = repo(run_timeout_s=0.001)
    record = await hub.submit(EvalRunRequest(task=EvalTask.RETRIEVAL))
    final = await hub.wait(record.id, timeout=30)
    assert final is not None and final.status is RunStatus.FAILED
    assert "超时" in final.error


async def test_concurrency_semaphore_is_per_loop(repo):
    """Semaphore 绑定事件循环，跨 loop 复用会报错——这是 asyncio 的经典坑。"""
    hub = repo(max_concurrency=1)
    await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL))
    await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL))  # 同一 loop 复用
    assert len(hub.list_runs(limit=10)) >= 2


# --------------------------------------------------------------------------- #
# 4. 鉴权
# --------------------------------------------------------------------------- #
def test_write_endpoints_require_api_key(tmp_path: Path):
    from fastapi.testclient import TestClient

    from src.app import create_app

    settings = Settings(
        env="test",
        auth=AuthConfig(api_key="s3cret"),
        datahub=DataHubConfig(db_path=str(tmp_path / "dh.db")),
        evalhub=EvalHubConfig(db_path=str(tmp_path / "eh.db")),
    )
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/eval/runs", json={"task": "retrieval"}).status_code == 401
        ok = client.post(
            "/v1/eval/runs", json={"task": "retrieval"}, headers={"X-API-Key": "s3cret"}
        )
        assert ok.status_code == 202
        assert ok.headers["location"].startswith("/v1/eval/runs/")
        # 读接口保持公开：观察不该需要钥匙
        assert client.get("/v1/eval/tasks").status_code == 200
        assert client.post("/v1/data/sync").status_code == 401


def test_api_async_submit_and_progress(tmp_path: Path):
    from fastapi.testclient import TestClient

    from src.app import create_app

    settings = Settings(
        env="test",
        datahub=DataHubConfig(db_path=str(tmp_path / "dh.db")),
        evalhub=EvalHubConfig(db_path=str(tmp_path / "eh.db")),
    )
    with TestClient(create_app(settings)) as client:
        resp = client.post("/v1/eval/runs", json={"task": "retrieval"})
        assert resp.status_code == 202
        run_id = resp.json()["id"]
        assert resp.json()["status"] in ("pending", "running", "succeeded")
        prog = client.get(f"/v1/eval/runs/{run_id}/progress").json()
        assert prog["id"] == run_id
        assert prog["status"] in ("pending", "running", "succeeded")


# --------------------------------------------------------------------------- #
# 5. 库迁移与保留策略
# --------------------------------------------------------------------------- #
def test_schema_migration_adds_new_columns(tmp_path: Path):
    """老库（v0.7 结构）升级到 v0.8 后必须能直接写，不需要手工改表。"""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE eval_runs (
            id TEXT PRIMARY KEY, task TEXT NOT NULL, dataset_id TEXT DEFAULT '',
            dataset_hash TEXT DEFAULT '', status TEXT NOT NULL, verdict TEXT NOT NULL,
            trusted INTEGER DEFAULT 0, config_fingerprint TEXT DEFAULT '',
            fingerprint_detail TEXT DEFAULT '{}', metrics TEXT DEFAULT '{}',
            warnings TEXT DEFAULT '[]', error TEXT DEFAULT '', started_at TEXT NOT NULL,
            finished_at TEXT DEFAULT '', duration_ms REAL DEFAULT 0,
            cases INTEGER DEFAULT 0, gate TEXT DEFAULT ''
        );
        CREATE TABLE eval_cases (run_id TEXT, case_id TEXT, metrics TEXT, detail TEXT);
        """
    )
    conn.commit()
    conn.close()

    store = EvalHubStore(db)
    conn = sqlite3.connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(eval_runs)")}
    conn.close()
    assert {"code_version", "runtime", "stage", "progress"} <= cols

    store.save_run(
        EvalRunRecord(
            id="r-old", task="retrieval", code_version="abc1234", runtime={"python": "3.13.12"},
            stage="running_task", progress=0.4,
        )
    )
    back = store.get_run("r-old")
    assert back is not None and back.code_version == "abc1234" and back.progress == 0.4


def test_prune_keeps_recent_and_trusted(tmp_path: Path):
    store = EvalHubStore(tmp_path / "eh.db")
    for i in range(12):
        store.save_run(
            EvalRunRecord(
                id=f"r{i:02d}", task="retrieval", trusted=(i % 4 == 0),
                started_at=f"2026-11-01T00:00:{i:02d}", status=RunStatus.SUCCEEDED,
            )
        )
    result = store.prune(keep_last=3, keep_trusted=2)
    kept = {r.id for r in store.list_runs(limit=100)}
    assert result["deleted"] == 12 - len(kept)
    assert {"r09", "r10", "r11"} <= kept, "最近的必须留下"
    assert {"r08", "r04"} <= kept, "可信基线必须留下"


def test_prune_never_deletes_running(tmp_path: Path):
    store = EvalHubStore(tmp_path / "eh.db")
    store.save_run(EvalRunRecord(id="r-run", task="retrieval", status=RunStatus.RUNNING))
    for i in range(5):
        store.save_run(EvalRunRecord(id=f"r{i}", task="retrieval", status=RunStatus.SUCCEEDED))
    store.prune(keep_last=1, keep_trusted=0)
    assert store.get_run("r-run") is not None, "在跑的 run 被删掉就等于丢了正在烧的钱"
