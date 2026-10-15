# -*- coding: utf-8 -*-
"""一次性扩评测集脚本：30 -> 60 条。

原来 22 literal / 8 semantic，semantic 太少导致评测集主要靠字面题区分策略。
本次 +8 literal / +22 semantic，最终 30/30。semantic 题刻意避开原题字面用词，
只有语义检索能命中——这是评测集区分度的来源。
跑完即弃；校验所有 relevant id 必须存在于题库。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "config" / "retrieval_eval.json"
BANK = ROOT / "config" / "question_bank.json"

NEW_CASES = [
    # ---------------- literal +8 ----------------
    {"query": "系统提示词 写法 指令顺序 长度", "relevant": ["llm-09"], "note": "prompt 工程", "kind": "literal"},
    {"query": "token 成本估算 上下文占比", "relevant": ["llm-16", "llm-08"], "note": "成本", "kind": "literal"},
    {"query": "asyncio 会话存储 加锁 并发", "relevant": ["eng-10", "llm-12"], "note": "并发状态", "kind": "literal"},
    {"query": "P95 延迟升高 排查顺序", "relevant": ["eng-13"], "note": "性能排查", "kind": "literal"},
    {"query": "kappa paradox 类别不均衡 一致性", "relevant": ["data-09"], "note": "kappa 悖论", "kind": "literal"},
    {"query": "多重检验校正 BH Bonferroni 区别", "relevant": ["data-01"], "note": "统计校正", "kind": "literal"},
    {"query": "embedding 模型升级 向量索引重建", "relevant": ["data-12"], "note": "索引重建", "kind": "literal"},
    {"query": "三年职业规划 为什么选这个岗位", "relevant": ["beh-09"], "note": "规划", "kind": "literal"},
    # ---------------- semantic +22（刻意避开原题字面用词） ----------------
    {"query": "面试官让我证明某个改进真的有用，不能凭感觉", "relevant": ["proj-02", "proj-13"], "note": "消融与实测", "kind": "semantic"},
    {"query": "怎么知道搜出来的内容是真相关而不是分数虚高", "relevant": ["proj-11"], "note": "阈值标定", "kind": "semantic"},
    {"query": "模型一本正经地胡说八道怎么办", "relevant": ["llm-06"], "note": "幻觉", "kind": "semantic"},
    {"query": "对话轮数太多塞不下上下文了怎么办", "relevant": ["llm-05"], "note": "上下文压缩", "kind": "semantic"},
    {"query": "机器人反复做同一个动作停不下来", "relevant": ["llm-04"], "note": "死循环", "kind": "semantic"},
    {"query": "页面上的回答一个字一个字往外蹦是怎么做到的", "relevant": ["llm-19", "llm-03"], "note": "流式输出", "kind": "semantic"},
    {"query": "同一句话换个说法就搜不到了怎么解决", "relevant": ["proj-12"], "note": "中文分词", "kind": "semantic"},
    {"query": "付费接口突然不给用了系统怎么不崩", "relevant": ["llm-14"], "note": "降级", "kind": "semantic"},
    {"query": "机器裁判和真人裁判打分不一致怎么量化差距", "relevant": ["llm-15", "llm-07"], "note": "judge 一致性", "kind": "semantic"},
    {"query": "上线一段时间后用户问的问题类型慢慢变了，怎么发现", "relevant": ["data-04"], "note": "分布漂移", "kind": "semantic"},
    {"query": "考题混进了练习题里，成绩就不算数了", "relevant": ["data-07", "data-05"], "note": "数据泄漏", "kind": "semantic"},
    {"query": "班里九十九个健康人一个病人，全猜健康也有九十九分", "relevant": ["data-02"], "note": "不平衡指标", "kind": "semantic"},
    {"query": "对方服务器偶尔发脾气拒绝请求，客户端怎么办", "relevant": ["eng-03", "eng-16"], "note": "429 处理", "kind": "semantic"},
    {"query": "一个本地脚本想变成团队都能调的网络接口", "relevant": ["eng-01"], "note": "脚本到服务", "kind": "semantic"},
    {"query": "几百个请求一起发出去怎么不把自己打挂", "relevant": ["eng-04"], "note": "并发控制", "kind": "semantic"},
    {"query": "升级了翻译模型之后，之前存的指纹还有效吗", "relevant": ["data-12"], "note": "向量兼容性", "kind": "semantic"},
    {"query": "别人改过的配置被我一保存就覆盖了", "relevant": ["eng-06"], "note": "配置冲突", "kind": "semantic"},
    {"query": "给领导汇报模型效果，光报数字被打回来了", "relevant": ["data-08"], "note": "可视化沟通", "kind": "semantic"},
    {"query": "标注人手不够只能测几十条，这笔预算怎么花", "relevant": ["data-10"], "note": "标注预算", "kind": "semantic"},
    {"query": "快到截止日期了发现最重要的指标不达标", "relevant": ["beh-10"], "note": "抗压", "kind": "semantic"},
    {"query": "上级坚持的方案我验证过是错的怎么办", "relevant": ["beh-12"], "note": "冲突", "kind": "semantic"},
    {"query": "工作里遇到完全没接触过的领域怎么上手", "relevant": ["beh-08"], "note": "学习路径", "kind": "semantic"},
]


def main() -> None:
    eval_set = json.loads(EVAL.read_text(encoding="utf-8"))
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    ids = {i["id"] for i in bank["items"]}
    bad = [c["query"] for c in NEW_CASES for r in c["relevant"] if r not in ids]
    if bad:
        raise SystemExit(f"relevant id 不在题库: {bad}")
    existing_q = {c["query"] for c in eval_set["cases"]}
    dup = [c["query"] for c in NEW_CASES if c["query"] in existing_q]
    if dup:
        raise SystemExit(f"query 重复: {dup}")
    eval_set["cases"].extend(NEW_CASES)
    EVAL.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from collections import Counter

    print(f"总数 {len(eval_set['cases'])} | {dict(Counter(c.get('kind') for c in eval_set['cases']))}")


if __name__ == "__main__":
    main()
