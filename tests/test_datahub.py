# -*- coding: utf-8 -*-
"""数据中台测试。

重点不是"能跑"，而是**门禁真的会拦住坏数据**：
    · 评测集指向不存在的题号（引用断裂）→ 必须 blocked，而不是悄悄跑出低分；
    · blocked 不能顶掉上一个可用版本；
    · 同内容重复同步必须是 unchanged（幂等），否则每次启动都产生新版本，版本号失去意义。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.config import DataHubConfig, Settings
from src.datahub import DataHub, content_hash
from src.datahub.quality import check, normalize
from src.datahub.models import DatasetKind

BANK = {
    "version": "test",
    "items": [
        {"id": "q1", "dimension": "项目深挖", "difficulty": 2, "tags": ["RAG"], "question": "你的 RAG 项目怎么切分文档？"},
        {"id": "q2", "dimension": "工程素养", "difficulty": 3, "tags": ["缓存"], "question": "如何设计多级缓存？"},
        {"id": "q3", "dimension": "算法基础", "difficulty": 1, "tags": ["排序"], "question": "快速排序的时间复杂度？"},
    ],
}

RETRIEVAL = {
    "version": "test",
    "cases": [
        {"query": "RAG 文档切分", "relevant": ["q1"], "kind": "literal"},
        {"query": "缓存设计", "relevant": ["q2"], "kind": "semantic"},
    ],
}

EVAL_SET = {
    "name": "test-eval",
    "version": "0.1",
    "rubric_version": "v1",
    "cases": [
        {
            "id": "c1",
            "dimension": "追问合理性",
            "dialogue": [{"role": "user", "content": "我做了 RAG 项目"}],
            "reference": "应追问切分策略",
            "expects_refusal": False,
            "human_score": 4,
        }
    ],
}

BENCH = [{"id": "b1", "note": "开场", "text": "你好，可以开始了吗？"}]


@pytest.fixture()
def mini_repo(tmp_path: Path):
    root = tmp_path / "repo"
    cfg = root / "config"
    cfg.mkdir(parents=True)
    (cfg / "question_bank.json").write_text(json.dumps(BANK, ensure_ascii=False), encoding="utf-8")
    (cfg / "retrieval_eval.json").write_text(json.dumps(RETRIEVAL, ensure_ascii=False), encoding="utf-8")
    (cfg / "eval_set.example.json").write_text(json.dumps(EVAL_SET, ensure_ascii=False), encoding="utf-8")
    (cfg / "bench_questions.json").write_text(json.dumps(BENCH, ensure_ascii=False), encoding="utf-8")

    settings = Settings(env="test", datahub=DataHubConfig(db_path=str(tmp_path / "datahub.db")))
    hub = DataHub(settings, root=root)
    return root, hub


def write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 版本与幂等
# --------------------------------------------------------------------------- #
def test_sync_all_registers_every_dataset(mini_repo):
    _, hub = mini_repo
    results = hub.sync_all()
    assert {r.dataset_id for r in results} == {"question_bank", "retrieval_eval", "eval_set", "bench_questions"}
    assert all(r.status == "registered" for r in results)
    assert all(r.quality is not None and r.quality.passed for r in results)


def test_sync_is_idempotent(mini_repo):
    _, hub = mini_repo
    hub.sync_all()
    again = hub.sync_all()
    assert all(r.status == "unchanged" for r in again), "同内容重复同步不应产生新版本"
    assert all(hub.get(r.dataset_id).versions == 1 for r in again)


def test_content_hash_changes_with_content(mini_repo):
    root, hub = mini_repo
    first = hub.sync("question_bank")
    write(root / "config/question_bank.json", {**BANK, "items": BANK["items"][:2]})
    second = hub.sync("question_bank")
    assert first.content_hash != second.content_hash
    assert second.status == "registered"
    assert hub.get("question_bank").versions == 2


def test_dirty_detected_when_file_changed_without_sync(mini_repo):
    root, hub = mini_repo
    hub.sync_all()
    assert hub.get("question_bank").dirty is False
    write(root / "config/question_bank.json", {**BANK, "items": BANK["items"][:2]})
    assert hub.get("question_bank").dirty is True, "改了文件没同步必须能被发现"


# --------------------------------------------------------------------------- #
# 质量门禁
# --------------------------------------------------------------------------- #
def test_dangling_reference_blocks_dataset(mini_repo):
    root, hub = mini_repo
    hub.sync("question_bank")
    bad = {"version": "t", "cases": [{"query": "不存在的题", "relevant": ["q999"], "kind": "literal"}]}
    write(root / "config/retrieval_eval.json", bad)

    result = hub.sync("retrieval_eval")
    assert result.status == "blocked"
    codes = [e.code for e in result.quality.errors]
    assert "dangling_reference" in codes
    assert hub.get("retrieval_eval").latest is None, "被拦截的数据集不应出现可用版本"


def test_blocked_does_not_replace_previous_version(mini_repo):
    root, hub = mini_repo
    hub.sync("question_bank")
    good = hub.sync("retrieval_eval")
    assert good.status == "registered"

    bad = {"version": "t", "cases": [{"query": "坏数据", "relevant": ["q999"], "kind": "literal"}]}
    write(root / "config/retrieval_eval.json", bad)
    blocked = hub.sync("retrieval_eval")

    assert blocked.status == "blocked"
    latest = hub.get("retrieval_eval").latest
    assert latest is not None and latest.content_hash == good.content_hash, "坏数据不得顶掉上一个可用版本"


def test_duplicate_query_is_error():
    report = check(
        "retrieval_eval",
        DatasetKind.RETRIEVAL_EVAL,
        "hash",
        {"cases": [
            {"query": "RAG 切分", "relevant": ["q1"], "kind": "literal"},
            {"query": "RAG  切分", "relevant": ["q2"], "kind": "literal"},
        ]},
    )
    assert not report.passed
    assert any(e.code == "duplicate_query" for e in report.errors)


def test_last_turn_must_be_user():
    report = check(
        "eval_set",
        DatasetKind.EVAL_SET,
        "hash",
        {"cases": [{
            "id": "c1",
            "dimension": "追问合理性",
            "dialogue": [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "回答"},
            ],
        }]},
    )
    assert not report.passed
    assert any(e.code == "last_turn_not_user" for e in report.errors)


def test_dimension_gap_is_warning_not_error():
    report = check("eval_set", DatasetKind.EVAL_SET, "hash", EVAL_SET)
    assert report.passed
    assert any(w.code == "dimension_gap" for w in report.warnings)


def test_bad_difficulty_is_warning():
    data = {"items": [{"id": "q1", "dimension": "d", "difficulty": 9, "tags": ["t"], "question": "q"}]}
    report = check("question_bank", DatasetKind.QUESTION_BANK, "hash", data)
    assert report.passed
    assert any(w.code == "bad_difficulty" for w in report.warnings)


def test_normalize_strips_punctuation():
    assert normalize("RAG 切分，怎么做？") == normalize("RAG切分怎么做")


def test_content_hash_is_stable_and_short():
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")
    assert len(content_hash(b"abc")) == 16


# --------------------------------------------------------------------------- #
# 血缘与快照
# --------------------------------------------------------------------------- #
def test_lineage_contains_dependency(mini_repo):
    _, hub = mini_repo
    hub.sync_all()
    graph = hub.lineage()
    assert any(e.from_dataset == "retrieval_eval" and e.to_dataset == "question_bank" for e in graph.edges)
    assert hub.get("question_bank").referenced_by == ["retrieval_eval"]


def test_snapshot_reports_untrusted_when_dirty(mini_repo):
    root, hub = mini_repo
    hub.sync_all()
    assert hub.snapshot("question_bank")["trusted_data"] is True

    write(root / "config/question_bank.json", {**BANK, "items": BANK["items"][:2]})
    snap = hub.snapshot("question_bank")
    assert snap["dirty"] is True
    assert snap["trusted_data"] is False, "磁盘与登记不一致时数据不可信"


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_api_lists_datasets(client):
    resp = client.get("/v1/data/datasets")
    assert resp.status_code == 200
    ids = {d["spec"]["id"] for d in resp.json()}
    assert {"question_bank", "retrieval_eval", "eval_set", "bench_questions"} <= ids


def test_api_dataset_detail_and_404(client):
    assert client.get("/v1/data/datasets/question_bank").status_code == 200
    assert client.get("/v1/data/datasets/not_exist").status_code == 404


def test_api_quality_endpoint(client):
    resp = client.get("/v1/data/datasets/question_bank/quality")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == "question_bank"
    assert body["errors"] == []


def test_api_lineage(client):
    resp = client.get("/v1/data/lineage")
    assert resp.status_code == 200
    assert "retrieval_eval" in resp.json()["nodes"]


def test_api_sync_is_idempotent(client):
    first = client.post("/v1/data/sync").json()
    second = client.post("/v1/data/sync").json()
    assert all(r["status"] == "unchanged" for r in second)
    assert len(first) == len(second)


def test_health_reports_platforms_enabled(client):
    body = client.get("/health").json()
    assert body["datahub_enabled"] is True
    assert body["evalhub_enabled"] is True
