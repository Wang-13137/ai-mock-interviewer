# -*- coding: utf-8 -*-
"""在同一份标注上对比三种检索策略，产出 Markdown + JSON 报告。

用法：

    python scripts/eval_retrieval.py                 # bm25 + hybrid(hashing 替身)
    python scripts/eval_retrieval.py --embed bge     # 装了 sentence-transformers 后，跑真实语义 embedding
    python scripts/eval_retrieval.py --strategies bm25

⚠️ 只要参与对比的策略里**有任何一路用的是非语义嵌入**（HashingEmbedder），
报告顶部就会自动打上替代物警示块——这些数字只能证明链路通，不得对外引用。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.available import available_backends  # noqa: E402
from src.rag.corpus import load_bank  # noqa: E402
from src.rag.evaluate import (  # noqa: E402
    discriminative_power,
    evaluate,
    load_cases,
    render_report,
)
from src.rag.retriever import Retriever, Strategy  # noqa: E402

KS = (1, 3, 5)

MOCK_WARNING = (
    "> ## ⚠️ 本报告包含非语义嵌入（替身）\n"
    ">\n"
    "> 参与对比的策略中，向量路由 `HashingEmbedder`（词袋哈希投影）而非真实语义模型。\n"
    "> 它的语义相似度是**假的**——「批处理」和「跑批」在它眼里毫无关系。\n"
    ">\n"
    "> **这些数字只能证明检索链路、融合逻辑和评测代码是对的。\n"
    "> 一律不得写进简历、面经、README 效果表或任何对外材料。**\n"
    ">\n"
    "> 另：语料只有 30 条题目，Recall@k 的绝对值没有参考价值，\n"
    "> 只看「同一份标注上，哪个策略相对更好」这一个结论。\n"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="检索策略对比评测")
    ap.add_argument("--bank", default="config/question_bank.json")
    ap.add_argument("--cases", default="config/retrieval_eval.json")
    ap.add_argument(
        "--strategies",
        default="bm25,vector,hybrid",
        help="逗号分隔：bm25 / vector / hybrid",
    )
    ap.add_argument(
        "--embed",
        default="auto",
        help="auto | hashing | sentence-transformers（auto：有库就用，没有就用替身）",
    )
    ap.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="embedding 模型（Hub id 或本地路径；Hub id 优先解析 models/ 本地镜像）",
    )
    ap.add_argument("--tag", default="retrieval")
    args = ap.parse_args()

    bank = load_bank(args.bank)
    cases = load_cases(args.cases)
    if not bank.get("items"):
        print("题库为空，无法评测", file=sys.stderr)
        return 2
    print(f"语料 {len(bank['items'])} 条 | 评测 query {len(cases)} 条 | 后端探测 {available_backends()}")

    reports = []
    for raw in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        strategy = Strategy(raw)
        retriever = Retriever.from_bank(
            bank, strategy=strategy, embedder_name=args.embed, model_name=args.model
        )
        reports.append(evaluate(retriever, cases, KS))
        meta = retriever.describe()
        print(
            f"  {strategy.value:<7} Recall@3={reports[-1]['recall@3']:.3f} "
            f"MRR@10={reports[-1]['mrr@10']:.3f} Hit@1={reports[-1]['hit@1']:.3f} "
            f"embedder={meta['embedder'] or '—'} semantic={meta['semantic_embedding']}"
        )

    tainted = [r for r in reports if not r["semantic"] and r["strategy"] in ("vector", "hybrid")]
    note = ""
    diag = discriminative_power(reports)
    if diag["useless"]:
        note = (
            "⚠️ **本评测集当前没有区分度**：所有策略 MRR 都接近满分，"
            "选策略时它给不出信息。需要增加更多 semantic 类（字面不重叠）的 query。"
        )

    md = render_report(reports, KS, note=note)
    if tainted:
        md = MOCK_WARNING + "\n" + md

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"{args.tag}_{ts}.md"
    js_path = out_dir / f"{args.tag}_{ts}.json"
    md_path.write_text(md, encoding="utf-8")
    js_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n区分度：mrr_spread={diag['mrr_spread']} 饱和={diag['saturated']} "
        f"semantic 子集 MRR={diag['semantic_subset_mrr']}"
    )
    if diag["useless"]:
        print("⚠️ 评测集无区分度，结论不可用。")
    print(f"\n报告：{md_path}\n      {js_path}")
    if tainted:
        print("\n⚠️ 含替身嵌入，数字不得对外引用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
