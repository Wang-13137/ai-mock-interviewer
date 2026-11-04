# -*- coding: utf-8 -*-
"""回收人工标注表：校验后把 human_score 写回 eval_set。

用法：
    python scripts/import_human_labels.py docs/labeling/eval-set-标注表.csv

校验规则（不合格直接退出码 1，绝不静默丢弃）：
    - id 必须存在于 eval_set；
    - human_score 必须是 1-5 的整数（允许留空 = 暂不标，但会报告剩余量）；
    - 同一 id 出现两次直接报错（Excel 复制粘贴的常见事故）。

写回之后建议的下一步：
    python -m src.cli data sync                  # 新版本登记 + 质量门禁
    python -m src.cli evals run --task generation --baseline latest
    # run 的逐条明细里会带上 human_score，报告直接给出 kappa / QWK
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_SET = REPO / "config" / "eval_set.example.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="export_labeling_sheet.py 产出并填好的标注表")
    ap.add_argument("--dry-run", action="store_true", help="只校验不写回")
    args = ap.parse_args()

    src = Path(args.csv_path)
    if not src.exists():
        print(f"[FAIL] 找不到 {src}")
        return 1

    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in data["cases"]}

    rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
    errors: list[str] = []
    updates: dict[str, int] = {}
    seen: set[str] = set()

    for i, row in enumerate(rows):
        cid = (row.get("id") or "").strip()
        where = f"第 {i + 2} 行"  # CSV 首行是表头
        if cid not in by_id:
            errors.append(f"{where}: id={cid!r} 不在 eval_set 里（是不是导出后又改了评测集？重新导出再标）")
            continue
        if cid in seen:
            errors.append(f"{where}: id={cid} 重复出现")
            continue
        seen.add(cid)
        raw = (row.get("human_score") or "").strip()
        if raw == "":
            continue  # 留空 = 未标，不算错
        try:
            score = int(float(raw))
        except ValueError:
            errors.append(f"{where}: id={cid} 的 human_score={raw!r} 不是数字")
            continue
        if not 1 <= score <= 5:
            errors.append(f"{where}: id={cid} 的 human_score={score} 超出 1-5")
            continue
        updates[cid] = score

    if errors:
        print(f"[FAIL] {len(errors)} 个问题：")
        for e in errors:
            print(f"  - {e}")
        return 1

    if args.dry_run:
        print(f"[dry-run] {len(updates)} 条可写回，全部校验通过")
        return 0

    for cid, score in updates.items():
        by_id[cid]["human_score"] = score

    scored = sum(1 for c in data["cases"] if c.get("human_score") is not None)
    EVAL_SET.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写回 {len(updates)} 条；总进度 {scored}/{len(data['cases'])}")
    if scored < 30:
        print(f"还差 {30 - scored} 条到 n=30——不到 30 条别引用 kappa/QWK（kappa paradox）。")
    else:
        print("n≥30，一致性数字可外引。下一步：")
        print("  python -m src.cli data sync")
        print("  python -m src.cli evals run --task generation --baseline latest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
