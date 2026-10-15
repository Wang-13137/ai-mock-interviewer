# -*- coding: utf-8 -*-
"""一次性扩库脚本：题库 32 -> 80。

新增 48 条按维度补齐（项目深挖+10 / LLM 工程+12 / 工程基础+10 / 数据与统计+8 / 行为与动机+8），
并特意补了 7 条 difficulty=1（原来 32 条里只有 1 条，难度分布失真）。
跑完即弃，不保留进 CI；如需再扩，直接改 NEW_ITEMS 重跑。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "config" / "question_bank.json"

NEW_ITEMS = [
    # ---------------- 项目深挖 +10 ----------------
    {"id": "proj-09", "dimension": "项目深挖", "difficulty": 2, "tags": ["RAG", "混合检索"],
     "question": "混合检索里 BM25 和向量检索的分数尺度完全不同，你是怎么把两路结果合到一起的？"},
    {"id": "proj-10", "dimension": "项目深挖", "difficulty": 2, "tags": ["评测", "饱和"],
     "question": "你的检索评测集只有几十条标注，怎么防止「评测集饱和」——所有方法都接近满分、看不出差异？"},
    {"id": "proj-11", "dimension": "项目深挖", "difficulty": 3, "tags": ["阈值", "标定"],
     "question": "向量检索的相似度阈值是怎么定的？换一个 embedding 模型之后，这个阈值还有效吗？"},
    {"id": "proj-12", "dimension": "项目深挖", "difficulty": 2, "tags": ["分词", "中文"],
     "question": "中文检索里直接用单字分词会有什么问题？你是怎么处理的？"},
    {"id": "proj-13", "dimension": "项目深挖", "difficulty": 3, "tags": ["负结果", "决策"],
     "question": "有没有哪个「标准做法」你在项目里实测后发现是负收益？你当时怎么验证、怎么决策的？"},
    {"id": "proj-14", "dimension": "项目深挖", "difficulty": 2, "tags": ["离线", "测试"],
     "question": "没有 API Key、没有 GPU 的开发机上，你怎么让整条 RAG 链路还能跑起来、还能被测试覆盖？"},
    {"id": "proj-15", "dimension": "项目深挖", "difficulty": 3, "tags": ["幻觉", "归因"],
     "question": "引用精确率卡住上不去时，你怎么区分是检索召回的问题还是生成环节的问题？"},
    {"id": "proj-16", "dimension": "项目深挖", "difficulty": 2, "tags": ["毕设", "VR"],
     "question": "你的毕设 VR 系统里，AI 模块和三维引擎之间是怎么通信的？延迟怎么控制？"},
    {"id": "proj-17", "dimension": "项目深挖", "difficulty": 3, "tags": ["质谱", "选型"],
     "question": "MALDI 质谱数据维度高、样本少，你为什么先用传统机器学习而不是直接上深度学习？"},
    {"id": "proj-18", "dimension": "项目深挖", "difficulty": 2, "tags": ["复盘", "负结果"],
     "question": "你在项目文档里专门记录了没跑通的实验，为什么要把负结果也留下来？"},
    # ---------------- LLM 工程能力 +12 ----------------
    {"id": "llm-09", "dimension": "LLM 工程能力", "difficulty": 2, "tags": ["prompt"],
     "question": "系统提示词里你会写哪些内容？指令的顺序和长度对模型行为有影响吗？"},
    {"id": "llm-10", "dimension": "LLM 工程能力", "difficulty": 2, "tags": ["结构化输出", "JSON"],
     "question": "怎么让模型稳定输出合法 JSON？模型不遵守格式时你的兜底策略是什么？"},
    {"id": "llm-11", "dimension": "LLM 工程能力", "difficulty": 3, "tags": ["工具调用", "协议"],
     "question": "不用厂商原生 function calling，自己用提示词协议实现工具调用，好处和代价分别是什么？"},
    {"id": "llm-12", "dimension": "LLM 工程能力", "difficulty": 2, "tags": ["会话", "并发"],
     "question": "多轮面试里会话状态怎么管理？两个请求同时改同一个会话会出什么坑？"},
    {"id": "llm-13", "dimension": "LLM 工程能力", "difficulty": 3, "tags": ["缓存"],
     "question": "RAG 链路里哪些环节适合加缓存？embedding、检索结果、LLM 响应的失效策略分别怎么设计？"},
    {"id": "llm-14", "dimension": "LLM 工程能力", "difficulty": 2, "tags": ["降级", "可用性"],
     "question": "上游模型挂了或者额度用尽，你的服务怎么降级才不直接把错误甩给用户？"},
    {"id": "llm-15", "dimension": "LLM 工程能力", "difficulty": 3, "tags": ["评测", "judge"],
     "question": "LLM-as-judge 和人工评分的一致性用什么指标衡量？Cohen's kappa 在什么情况下会失效？"},
    {"id": "llm-16", "dimension": "LLM 工程能力", "difficulty": 2, "tags": ["成本", "token"],
     "question": "怎么估算一次多轮对话的 token 成本？上下文、工具回传内容各占多少比例？"},
    {"id": "llm-17", "dimension": "LLM 工程能力", "difficulty": 3, "tags": ["安全", "注入"],
     "question": "用户在对话里注入「忽略之前的指令」这类内容，你的系统有哪几道防线？"},
    {"id": "llm-18", "dimension": "LLM 工程能力", "difficulty": 2, "tags": ["多模型", "路由"],
     "question": "生成模型和裁判模型为什么不能用同一个？接多个厂商的模型时接口怎么统一？"},
    {"id": "llm-19", "dimension": "LLM 工程能力", "difficulty": 2, "tags": ["SSE", "协议"],
     "question": "SSE 和 WebSocket 的区别是什么？为什么 LLM 服务普遍用 SSE 做流式输出？"},
    {"id": "llm-20", "dimension": "LLM 工程能力", "difficulty": 1, "tags": ["采样", "基础"],
     "question": "temperature 和 top_p 分别控制什么？面试问答这种场景你会怎么设置？"},
    # ---------------- 工程基础 +10 ----------------
    {"id": "eng-07", "dimension": "工程基础", "difficulty": 1, "tags": ["git"],
     "question": "git 里 rebase 和 merge 的区别是什么？团队协作时你怎么选？"},
    {"id": "eng-08", "dimension": "工程基础", "difficulty": 2, "tags": ["配置", "密钥"],
     "question": "配置用 YAML 默认值加环境变量覆盖，这套机制怎么实现？API 密钥应该怎么管？"},
    {"id": "eng-09", "dimension": "工程基础", "difficulty": 2, "tags": ["日志", "埋点"],
     "question": "一次请求从进来到返回，你会埋哪些点？TTFT 和端到端时延为什么要分开记录？"},
    {"id": "eng-10", "dimension": "工程基础", "difficulty": 3, "tags": ["asyncio", "锁"],
     "question": "asyncio 里共享的会话存储为什么还要加锁？不加锁会出现什么现象？"},
    {"id": "eng-11", "dimension": "工程基础", "difficulty": 2, "tags": ["测试"],
     "question": "评测集和单元测试的分工是什么？什么样的 bug 值得固化成回归测试？"},
    {"id": "eng-12", "dimension": "工程基础", "difficulty": 2, "tags": ["依赖", "降级"],
     "question": "引入一个 1GB 的模型依赖之前你会考虑哪些因素？怎么设计优雅降级？"},
    {"id": "eng-13", "dimension": "工程基础", "difficulty": 3, "tags": ["性能", "排查"],
     "question": "服务 P95 延迟突然升高，你的排查顺序是什么？"},
    {"id": "eng-14", "dimension": "工程基础", "difficulty": 2, "tags": ["API", "兼容"],
     "question": "接口版本兼容怎么做？「标记废弃」和「直接删字段」的区别是什么？"},
    {"id": "eng-15", "dimension": "工程基础", "difficulty": 2, "tags": ["部署", "环境"],
     "question": "开发机、评测机、线上环境的差异怎么管理？怎么避免「在我机器上能跑」？"},
    {"id": "eng-16", "dimension": "工程基础", "difficulty": 1, "tags": ["HTTP", "基础"],
     "question": "HTTP 429 和 503 分别意味着什么？客户端的应对策略有什么不同？"},
    # ---------------- 数据与统计 +8 ----------------
    {"id": "data-06", "dimension": "数据与统计", "difficulty": 2, "tags": ["指标"],
     "question": "Precision、Recall、F1 分别回答什么问题？检索场景为什么更常用 Recall@k 和 MRR？"},
    {"id": "data-07", "dimension": "数据与统计", "difficulty": 3, "tags": ["泄漏"],
     "question": "数据泄漏有哪些隐蔽的形式？除了批次效应，你还能想到哪种？"},
    {"id": "data-08", "dimension": "数据与统计", "difficulty": 2, "tags": ["可视化", "沟通"],
     "question": "给非技术背景的人讲模型效果，你会画什么图？为什么不直接报一串数字？"},
    {"id": "data-09", "dimension": "数据与统计", "difficulty": 3, "tags": ["kappa", "一致性"],
     "question": "kappa paradox 是什么？为什么类别极不均衡时 Cohen's kappa 会趋近于 0？"},
    {"id": "data-10", "dimension": "数据与统计", "difficulty": 2, "tags": ["标注", "采样"],
     "question": "标注预算只够标 30 条评测数据，你怎么分配字面改写和语义改写的比例？"},
    {"id": "data-11", "dimension": "数据与统计", "difficulty": 2, "tags": ["特征", "可解释性"],
     "question": "特征重要性排序和模型可解释性是一回事吗？置换检验为什么更可靠？"},
    {"id": "data-12", "dimension": "数据与统计", "difficulty": 3, "tags": ["embedding", "漂移"],
     "question": "embedding 模型升级之后，线上已有的向量索引必须重建吗？为什么？"},
    {"id": "data-13", "dimension": "数据与统计", "difficulty": 1, "tags": ["分布", "基础"],
     "question": "均值、中位数、P95 分别适合描述什么分布？延迟数据为什么要看 P95？"},
    # ---------------- 行为与动机 +8 ----------------
    {"id": "beh-06", "dimension": "行为与动机", "difficulty": 1, "tags": ["动机"],
     "question": "如果入职后发现实际工作内容和想象的不一样，你会怎么办？"},
    {"id": "beh-07", "dimension": "行为与动机", "difficulty": 2, "tags": ["协作", "求助"],
     "question": "讲一次你主动向他人求助的经历，当时卡在哪、怎么开口的？"},
    {"id": "beh-08", "dimension": "行为与动机", "difficulty": 2, "tags": ["转行", "学习"],
     "question": "从生物医学工程转到 AI 方向，你的自学路径是什么？哪一步最难？"},
    {"id": "beh-09", "dimension": "行为与动机", "difficulty": 1, "tags": ["规划"],
     "question": "你未来三年的职业规划是什么？为什么第一步选这个岗位？"},
    {"id": "beh-10", "dimension": "行为与动机", "difficulty": 2, "tags": ["抗压"],
     "question": "项目 deadline 前发现核心指标不达标，你会怎么处理？"},
    {"id": "beh-11", "dimension": "行为与动机", "difficulty": 2, "tags": ["复盘", "习惯"],
     "question": "你有写工作日志或复盘的习惯吗？最近一次复盘改变了你什么做法？"},
    {"id": "beh-12", "dimension": "行为与动机", "difficulty": 3, "tags": ["冲突"],
     "question": "如果主管坚持一个你验证过是错的技术方案，你会怎么做？"},
    {"id": "beh-13", "dimension": "行为与动机", "difficulty": 1, "tags": ["收尾"],
     "question": "除了简历上的项目，你业余还做过哪些和技术有关的事？"},
]


def main() -> None:
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    existing = {i["id"] for i in bank["items"]}
    dup = [i["id"] for i in NEW_ITEMS if i["id"] in existing]
    if dup:
        raise SystemExit(f"id 冲突: {dup}")
    dims = {i["dimension"] for i in bank["items"]}
    bad = [i["id"] for i in NEW_ITEMS if i["dimension"] not in dims]
    if bad:
        raise SystemExit(f"维度不在已知集合: {bad}")
    bank["items"].extend(NEW_ITEMS)
    bank["version"] = "0.2-expanded"
    bank["note"] = (
        "80 条题库（32 种子 + 48 扩充，2026-10-16）。检索走 src/rag/（bm25/vector/hybrid 三策略），"
        "search_question_bank 工具契约不变。扩库后必须重跑 scripts/calibrate_min_score.py 与评测。"
    )
    BANK.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from collections import Counter

    c = Counter(i["dimension"] for i in bank["items"])
    d = Counter(i["difficulty"] for i in bank["items"])
    print(f"总数 {len(bank['items'])} | 维度 {dict(c)} | 难度 {dict(sorted(d.items()))}")


if __name__ == "__main__":
    main()
