# -*- coding: utf-8 -*-
"""评测中台测试。

两类断言：
    1. 门禁的判定逻辑（纯函数，用构造的 run 覆盖 pass / regressed / incomparable）；
    2. 端到端（真跑一次 BM25 检索评测，落库、取明细、与基线比较）。

第 2 类里最关键的断言是 `trusted is True`：BM25 不涉及任何模型，数字天然真实；
换成 vector 且 embedding 退化时必须是 False——这条断言守住的是「替身数字不得外引」的红线。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.config import DataHubConfig, EvalHubConfig, Settings
from src.datahub import DataHub
from src.evalhub import EvalHub, EvalRunRecord, EvalRunRequest, EvalTask, RunStatus, Verdict
from src.evalhub.gate import compare, higher_is_better
from src.evalhub.runner import config_fingerprint

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


def make_run(
    run_id: str = "r1",
    task: str = "retrieval",
    metrics: dict | None = None,
    trusted: bool = True,
    fingerprint: str = "fp-a",
    dataset_hash: str = "hash-a",
) -> EvalRunRecord:
    return EvalRunRecord(
        id=run_id,
        task=task,
        metrics=metrics or {"mrr@10": 0.8, "recall@3": 0.9},
        trusted=trusted,
        config_fingerprint=fingerprint,
        dataset_hash=dataset_hash,
    )


@pytest.fixture()
def retrieval_repo(tmp_path: Path):
    root = tmp_path / "repo"
    cfg = root / "config"
    cfg.mkdir(parents=True)
    write(cfg / "question_bank.json", BANK)
    write(cfg / "retrieval_eval.json", RETRIEVAL)

    settings = Settings(
        env="test",
        datahub=DataHubConfig(db_path=str(tmp_path / "datahub.db")),
        evalhub=EvalHubConfig(db_path=str(tmp_path / "evalhub.db"), gate_threshold=0.02),
    )
    datahub = DataHub(settings, root=root)
    datahub.sync("question_bank")
    datahub.sync("retrieval_eval")
    hub = EvalHub(settings, datahub)
    return hub, datahub, root


# --------------------------------------------------------------------------- #
# 门禁：纯逻辑
# --------------------------------------------------------------------------- #
def test_gate_passes_when_identical():
    gate = compare(make_run("base"), make_run("cur"))
    assert gate.verdict is Verdict.PASS
    assert gate.regressed_metrics == []


def test_gate_regressed_when_metric_drops():
    base = make_run("base", metrics={"mrr@10": 0.80, "recall@3": 0.90})
    cur = make_run("cur", metrics={"mrr@10": 0.70, "recall@3": 0.90})
    gate = compare(base, cur, threshold=0.02)
    assert gate.verdict is Verdict.REGRESSED
    dropped = {d.metric for d in gate.regressed_metrics}
    assert dropped == {"mrr@10"}
    assert "mrr@10" in gate.reasons[0]


def test_gate_ignores_small_noise():
    base = make_run("base", metrics={"mrr@10": 0.800})
    cur = make_run("cur", metrics={"mrr@10": 0.795})  # 掉 0.005，小于阈值
    assert compare(base, cur, threshold=0.02).verdict is Verdict.PASS


def test_gate_ignores_relative_amplification_on_small_baseline():
    """小 baseline 的相对放大是噪声，不是退步。

    0.04 → 0.036 相对跌 10%，看着吓人，但在 60 条样本上只等于 1 条 case 翻转。
    v0.8 之前用「绝对 or 相对」任一超阈即判退步，于是这里会误报。
    现在要求两者同时超过，且（有明细时）CI 也要显著。
    """
    base = make_run("base", metrics={"hit@1": 0.04})
    cur = make_run("cur", metrics={"hit@1": 0.036})
    assert compare(base, cur, threshold=0.02, abs_threshold=0.02).verdict is Verdict.PASS


def test_gate_ignores_big_absolute_but_tiny_relative_drop():
    """反方向同理：baseline 很大时，绝对差看起来不小但相对只动了 0.3%。"""
    base = make_run("base", metrics={"avg_steps": 10.0})
    cur = make_run("cur", metrics={"avg_steps": 10.03})
    assert compare(base, cur, threshold=0.02, abs_threshold=0.02).verdict is Verdict.PASS


def test_gate_incomparable_on_task_mismatch():
    gate = compare(make_run("base", task="retrieval"), make_run("cur", task="generation"))
    assert gate.verdict is Verdict.INCOMPARABLE


def test_gate_incomparable_on_dataset_change():
    gate = compare(make_run("base", dataset_hash="h1"), make_run("cur", dataset_hash="h2"))
    assert gate.verdict is Verdict.INCOMPARABLE
    assert any("数据版本" in r for r in gate.reasons)


def test_gate_incomparable_on_config_change():
    gate = compare(make_run("base", fingerprint="fp-a"), make_run("cur", fingerprint="fp-b"))
    assert gate.verdict is Verdict.INCOMPARABLE
    assert any("配置指纹" in r for r in gate.reasons)


def test_gate_incomparable_when_baseline_untrusted():
    gate = compare(make_run("base", trusted=False), make_run("cur"))
    assert gate.verdict is Verdict.INCOMPARABLE
    assert any("替身" in r for r in gate.reasons)


def test_gate_lower_is_better_for_latency():
    base = make_run("base", metrics={"latency_ms": 100.0})
    cur = make_run("cur", metrics={"latency_ms": 150.0})
    gate = compare(base, cur, threshold=0.02)
    assert gate.verdict is Verdict.REGRESSED, "时延变大是退步"


def test_direction_detection():
    assert higher_is_better("mrr@10") is True
    assert higher_is_better("latency_ms") is False
    assert higher_is_better("avg_steps") is False


def test_config_fingerprint_is_stable_and_order_insensitive():
    a = config_fingerprint({"model": "glm-4-flash", "strategy": "bm25"})
    b = config_fingerprint({"strategy": "bm25", "model": "glm-4-flash"})
    c = config_fingerprint({"model": "glm-4-air", "strategy": "bm25"})
    assert a == b
    assert a != c


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
async def test_retrieval_run_is_trusted_and_persisted(retrieval_repo):
    hub, _, _ = retrieval_repo
    record = await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL))

    assert record.status is RunStatus.SUCCEEDED
    assert record.trusted is True, "BM25 不涉及模型，数字应被视为可信"
    assert record.metrics["mrr@10"] > 0
    assert record.dataset_hash, "必须绑定数据快照 hash"
    assert record.config_fingerprint

    stored = hub.get_run(record.id)
    assert stored is not None and stored.metrics["mrr@10"] == record.metrics["mrr@10"]
    cases = hub.get_cases(record.id)
    assert len(cases) == 3, "逐条明细必须落库，否则无法归因"


async def test_second_run_with_latest_baseline_passes(retrieval_repo):
    hub, _, _ = retrieval_repo
    first = await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL))
    second = await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL, baseline_run="latest"))

    assert second.gate is not None
    assert second.verdict is Verdict.PASS
    assert second.gate.baseline_run == first.id


async def test_gate_marks_incomparable_when_threshold_changed(retrieval_repo):
    """换配置（阈值）后应判不可比，而不是误报退步。"""
    hub, _, _ = retrieval_repo
    first = await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL, params={"top_k": 3}))
    second = await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL, params={"top_k": 5}, baseline_run=first.id))
    # top_k 参与指纹 -> 配置变了 -> 不可比
    assert second.verdict is Verdict.INCOMPARABLE


async def test_run_rejected_when_dataset_blocked(tmp_path: Path):
    root = tmp_path / "repo"
    cfg = root / "config"
    cfg.mkdir(parents=True)
    write(cfg / "question_bank.json", BANK)
    # 引用不存在的题号 -> 质量门禁拦截 -> 没有可用版本
    write(cfg / "retrieval_eval.json", {"cases": [{"query": "坏", "relevant": ["nope"], "kind": "literal"}]})

    settings = Settings(
        env="test",
        datahub=DataHubConfig(db_path=str(tmp_path / "dh.db")),
        evalhub=EvalHubConfig(db_path=str(tmp_path / "eh.db")),
    )
    datahub = DataHub(settings, root=root)
    hub = EvalHub(settings, datahub)

    record = await hub.run(EvalRunRequest(task=EvalTask.RETRIEVAL))
    assert record.status is RunStatus.REJECTED
    assert "质量门禁" in record.error or "未登记" in record.error
    assert not record.metrics, "被拒绝的 run 不应产出任何指标"


async def test_unknown_task_is_rejected(retrieval_repo):
    hub, _, _ = retrieval_repo
    with pytest.raises(ValueError):
        EvalTask("nonexistent")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_api_lists_tasks(client):
    body = client.get("/v1/eval/tasks").json()
    assert {t["name"] for t in body} == {"retrieval", "generation", "agent"}


def test_api_run_and_fetch(client):
    # wait=true：脚本/CI 场景要一次拿到结果，语义与 v0.7 之前一致
    resp = client.post("/v1/eval/runs?wait=true", json={"task": "retrieval"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["trusted"] is True
    assert body["metrics"]["mrr@10"] > 0

    run_id = body["id"]
    detail = client.get(f"/v1/eval/runs/{run_id}")
    assert detail.status_code == 200
    cases = client.get(f"/v1/eval/runs/{run_id}/cases")
    assert cases.status_code == 200 and len(cases.json()) > 0

    runs = client.get("/v1/eval/runs?task=retrieval").json()
    assert any(r["id"] == run_id for r in runs)


def test_api_gate_endpoint(client):
    first = client.post("/v1/eval/runs?wait=true", json={"task": "retrieval"}).json()
    second = client.post(
        "/v1/eval/runs?wait=true", json={"task": "retrieval", "baseline_run": first["id"]}
    ).json()
    assert second["verdict"] == "pass"

    gate = client.post(f"/v1/eval/runs/{second['id']}/gate?baseline={first['id']}").json()
    assert gate["verdict"] == "pass"


def test_api_run_404(client):
    assert client.get("/v1/eval/runs/not-exist").status_code == 404
