# -*- coding: utf-8 -*-
"""CI 回归门禁：跑一次 BM25 检索评测，与固化基线比较，退步就让 CI 变红。

为什么基线是文件而不是"库里 latest"：CI 每次都是全新的数据库，没有历史。
把基线 run 的指标 + 逐条明细固化进 `config/ci_baseline.json`（进 git），
可比较性由 compare() 的四道检查兜底——数据或配置变过会判 incomparable，
这时必须显式 `--update` 重新定基线并提交 diff，**基线永远不能被静默漂移**。

用法：
    python scripts/ci_gate.py              # CI 模式：跑评测 -> 与基线比 -> 退步 exit 1
    python scripts/ci_gate.py --update     # 显式重新定基线（review 后随代码一起提交）

只允许 BM25（零 LLM、零重依赖、CI 秒级）；语义路径的基线依赖模型权重，
不适合固化进 git，留在本地 evalhub 里走 `evals gate`。

退出码：0 = pass；1 = regressed / incomparable / 基线缺失。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_settings  # noqa: E402
from src.datahub import DataHub  # noqa: E402
from src.evalhub import EvalHub, EvalRunRequest, EvalTask  # noqa: E402
from src.evalhub.gate import compare, render_gate  # noqa: E402
from src.evalhub.models import CaseRow, EvalRunRecord, Verdict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO / "config" / "ci_baseline.json"
TASK = EvalTask.RETRIEVAL


def build_hub(tmp_db: Path) -> EvalHub:
    settings = load_settings()
    settings.evalhub.db_path = str(tmp_db / "ci_evalhub.db")
    settings.evalhub.gate_abs_threshold = 0.02
    settings.evalhub.gate_threshold = 0.02
    datahub = DataHub(settings, root=REPO)
    datahub.sync_all()
    return EvalHub(settings, datahub)


async def run_once() -> tuple[EvalRunRecord, list[CaseRow]]:
    with tempfile.TemporaryDirectory() as td:
        hub = build_hub(Path(td))
        record = await hub.run(EvalRunRequest(task=TASK))
        if record.status.value != "succeeded":
            print(f"[FAIL] 评测 run 失败：{record.error}")
            sys.exit(1)
        cases = list(hub.get_cases(record.id))
    return record, cases


def baseline_from_file(payload: dict) -> tuple[EvalRunRecord, list[CaseRow]]:
    b = payload["baseline"]
    record = EvalRunRecord(
        id=b.get("run_id", "ci-baseline"),
        task=payload["task"],
        dataset_hash=b.get("dataset_hash", ""),
        config_fingerprint=b.get("config_fingerprint", ""),
        metrics={k: float(v) for k, v in b["metrics"].items()},
        trusted=True,
    )
    cases = [
        CaseRow(case_id=c["case_id"], metrics=c["metrics"], detail={})
        for c in b.get("cases", [])
    ]
    return record, cases


async def main() -> int:
    ap = argparse.ArgumentParser(description="CI 回归门禁（retrieval / BM25）")
    ap.add_argument("--update", action="store_true", help="用本次 run 重新定基线（需人工 review 后提交）")
    args = ap.parse_args()

    record, cases = await run_once()
    print(f"本次 run：mrr@10={record.metrics.get('mrr@10')} recall@3={record.metrics.get('recall@3')} "
          f"trusted={record.trusted} fingerprint={record.config_fingerprint}")

    if not record.trusted:
        print("[FAIL] 本次 run 不可信（替身产出），CI 环境配置有问题——先修环境再谈门禁")
        return 1

    if args.update:
        payload = {
            "task": record.task,
            "updated_at": record.finished_at,
            "gate": {"abs_threshold": 0.02, "pct_threshold": 0.02, "bootstrap_iters": 1000},
            "baseline": {
                "run_id": record.id,
                "dataset_hash": record.dataset_hash,
                "config_fingerprint": record.config_fingerprint,
                "metrics": record.metrics,
                "cases": [{"case_id": c.case_id, "metrics": c.metrics} for c in cases],
            },
        }
        BASELINE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] 基线已写入 {BASELINE_PATH.relative_to(REPO)}——请 review 这份 diff 并随代码一起提交")
        return 0

    if not BASELINE_PATH.exists():
        print(f"[FAIL] 找不到基线文件 {BASELINE_PATH.relative_to(REPO)}——先在本地跑 python scripts/ci_gate.py --update 生成")
        return 1

    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    base, base_cases = baseline_from_file(payload)
    gate = compare(
        base,
        record,
        base_cases=base_cases or None,
        cur_cases=cases or None,
        abs_threshold=payload.get("gate", {}).get("abs_threshold", 0.02),
        pct_threshold=payload.get("gate", {}).get("pct_threshold", 0.02),
        iters=payload.get("gate", {}).get("bootstrap_iters", 1000),
    )
    print()
    print(render_gate(gate))

    if gate.verdict is Verdict.REGRESSED:
        print("\n[FAIL] 检索指标退步，阻断合并。修复退步，或（确认是预期改动时）--update 重新定基线并提交 diff。")
        return 1
    if gate.verdict is Verdict.INCOMPARABLE:
        print("\n[FAIL] 本次 run 与基线不可比。改了评测数据或检索配置后，需要显式 --update 重新定基线并提交 diff——基线不能被静默漂移。")
        return 1
    print("\n[OK] 未检测到回归")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
