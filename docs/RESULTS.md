# 对外数字真值表（Single Source of Truth）

> **本文件是本项目唯一权威的数字口径。** 任何对外材料——简历、博客、`README.md`、面经、面试口述——只能引用【A 可外引】与【D 带限定引用】两区。
> 判定日期：**2026-11-04**。依据：W8 博客终检「替身时延数字直接删掉不留」、W10 简历清除禁用数字的本人决断。
> **冲突处理：历史文档与本表不一致时，以本表为准**，不重新争论某次运行到底是真实还是替身。

---

## 判定规则（三条）

1. **可外引**：用真实语义嵌入（`semantic=true` + 本地 `bge-small-zh-v1.5` 权重）或真实云端 LLM 跑出、且原始报告可追溯到逐条明细。
2. **禁止外引**：`EchoClient` / `HashingEmbedder` / 离线抽取式摘要产生的任何数字（`src/rag/vector.py:4-15` 自己定义为「不得用于效果报告」）。
3. **口径存疑**：历史文档互相矛盾（如标注「真实」但 `.gitignore` 注「含 mock 数据」）→ **一律不外引，也不重新判定真实性**。

> 关于 `reports/`：该目录被 `.gitignore` 排除（注释写明「含 mock 数据，避免误提交到公开仓库」）。
> 因此**只有**经 `scripts/publish_evidence.py` 脱敏、提交进 `docs/evidence/` 的四份报告才是对外可验证证据。

---

## A. 可外引（有公开逐条明细）

| 数字 / 结论 | 口径 | 证据文件 |
| --- | --- | --- |
| 题库 80 条、检索评测集 60 条（literal 30 / semantic 30） | 自建手工标注 | `config/question_bank.json`、`config/retrieval_eval.json` |
| `bm25`　Recall@3 **0.792** / MRR@10 **0.767** / Hit@1 **0.717** | 最终 60 条集 | `docs/evidence/retrieval-bge-60q.{md,json}` |
| `vector`　**0.800** / **0.814** / **0.767** | 同上，真实 bge 嵌入 | 同上 |
| **`hybrid`　0.825 / 0.829 / 0.800** | 同上，真实 bge 嵌入 | 同上 |
| literal / semantic 分组 MRR：bm25 0.983/0.551、vector 1.000/0.628、hybrid 1.000/0.658 | 各 n=30 | 同上 |
| 重排在最终 60 条集上仍是负收益：vector MRR 0.814→**0.774**、hybrid 0.829→**0.767** | 同一份标注，唯一变量是重排开关 | `docs/evidence/retrieval-rerank-60q.{md,json}` |
| 重排在早期 30 条集上 MRR **0.883→0.828**（故 `rag.rerank=none` 默认关闭） | 早期集 22 literal + 8 semantic | `docs/evidence/retrieval-bge-80.{md,json}`、`retrieval-rerank-80.{md,json}` |
| 「结论随评测集难度翻转」：30 条简单集纯向量反超，补 22 条字面零重叠语义难例后 hybrid 反超，最终扩到 60 条 | 评测集构成，不是模型改动 | 上述四份报告按时间顺序 |
| 非负 IDF 变体、`min_score` 阈值、中文 bigram、RRF 融合 | 代码实现，非效果数字 | `src/rag/{bm25,vector,tokenize,fusion}.py` |
| pytest **139 passed** | **测试用例数量**，全部跑在离线替身 / `httpx.MockTransport` 上 | `tests/`、`.github/workflows/ci.yml` |

> 引用纪律：语料仅 80 条，**Recall@k 绝对值没有普适参考价值**，只能讲「同一份标注上哪个策略相对更好」。
> 「139 passed」只能说成「139 个自动化用例」，**不得说成「真实模型端到端验证」**。

---

## B. 禁止外引（替身 / 口径存疑）

| 数字 / 说法 | 原因 | 历史文档 |
| --- | --- | --- |
| TTFT P95 **8.70s**、端到端 P95 **9.18s**、P50 3.25s、720 次调用 | 原标注「真实 zhipu」与 `.gitignore`「含 mock 数据」及红线清单冲突；W8/W10 已删除 | `docs/bench-glm4flash.md`（**已改为历史归档·禁止外引**） |
| prompt token **3006→1107**、压缩比 63.2%、~17s 摘要代价、各轮 token 曲线 | 同上 | `docs/token-curve.md`（**已改为历史归档·禁止外引**） |
| 任何 `HashingEmbedder` 跑出的 Recall / MRR / Hit | 代码定义「不得用于效果报告」 | `src/rag/vector.py:4-15`、`scripts/eval_retrieval.py:MOCK_WARNING` |
| `EchoClient` 的 token / usage / 回复内容 | 离线替身 | `src/clients/llm.py:314-371` |
| 离线抽取式摘要的「压缩效果」 | 只保证链路可测 | `src/agent/memory.py:65-72` |
| 工具调用成功率 / 完成率 / 平均步数的具体数值 | `summarize_runs` 字段已实现，但**未做真实模型实测归档** | `README.md:147-148` |
| 「默认真实语义检索 / hybrid 已上线」 | 默认 `strategy: bm25`；语义依赖与权重不随仓库分发 | `config/default.yaml:52`、`requirements.txt` |
| 「v0.6 原生 function calling 已实测生效」作为**仓库证据** | 脚本里那句「✅ 已生效」只是条件字符串，CI 未接入、无输出归档 | `scripts/smoke_native.py:67` |
| `status=ok` 当作「上游健康」 | 恒为默认值，不探测上游 | `src/schemas/chat.py:73` |
| 「训练 / 微调模型」「生产级 / 会话持久化 / 高并发」 | 只调云端 API；状态是进程内 dict | 全仓 |

> 若面试官从旧版材料问到 B 区数字，标准答法：
> 「那组数据来自离线替身开发模式的基准，口径存疑，已从全部对外材料中移除；现在的口径以仓库 `docs/RESULTS.md` 为准。」

---

## C. 可口述、但仓库无归档（不能当"可验证证据"）

- **2026-11-04 本机真实 GLM-4-Flash 冒烟**：3 轮候选人对答，累计触发 2 次原生 function calling（`get_interview_progress`、`search_question_bank`），手写路径同轮 0 调用。真实跑通，但输出未归档、CI 未接入 → **只能是面试口述经历**。
- `/health` 自报 `semantic_retrieval=false`、评测脚本检测替身自动打印警告 → 这是**代码行为**，可直接指给面试官看，不算数字。
- `min_score` 0.30–0.40 平台区、0.60 时 semantic MRR 0.634→0.408 → `scripts/calibrate_min_score.py` 可重跑，但**无独立归档报告**；建议只讲方法，不把这两个数值当硬证据。

---

## D. 带限定才能引用

| 数字 | 必须同时说明的限定 |
| --- | --- |
| 「LLM-as-judge 五维评分，解析率 100%」 | 生成 `glm-4-flash`、裁判 `glm-4-air`，**n=10**；解析率只说明 JSON 输出可解析，**不等于评分可信**；同 provider 存在自评偏差警告；尚未做人工标注（≥30 条标完才谈一致性）。见 `reports/eval_20261016-015654.md`（本机，未公开）。 |
| 「裁判与生成分用不同模型」 | 模型不同但 **provider 相同**（均 zhipu），代码会主动打印自评偏差警告；不要说成「跨厂商避免自评」。 |
| 「AUC 0.92 / 0.917 基线」等 | **不属于本项目**，属 MALDI 系列项目，另有独立口径规则，本表不裁定。 |

---

## 证据索引与复现

```
docs/evidence/
├── retrieval-bge-60q.{md,json}       最终 60 条集 · 无重排（README 主表）
├── retrieval-rerank-60q.{md,json}    最终 60 条集 · 有重排（负结果）
├── retrieval-bge-80.{md,json}        早期 30 条集 · 无重排（0.883 出处）
└── retrieval-rerank-80.{md,json}     早期 30 条集 · 有重排（0.828 出处）
```

四份均含**逐条 query 明细**（query / 相关文档 ID / 召回结果 / RR），可逐格复算表格。
脱敏方式：仅把模型权重的本机绝对路径替换为模型名，**数值与明细未改动**，由 `scripts/publish_evidence.py` 生成并自检无路径残留。

复现一次新评测：

```bash
pip install -r requirements-semantic.txt        # 真实语义检索可选依赖
# 将 bge-small-zh-v1.5 权重放到 models/ 下（不随仓库分发）
python scripts/eval_retrieval.py --eval config/retrieval_eval.json --tag my-run
python scripts/publish_evidence.py              # 重新生成脱敏证据
```

> 注意：未装 `sentence-transformers` 时 `vector` / `hybrid` 会自动退化成 `HashingEmbedder`，
> 此时 `/health` 标 `semantic_retrieval=false`，跑出的数字**属于 B 区，一律不外引**。

---

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-10-16 | 检索评测四份原始报告生成于 `reports/`（本机，不公开）。 |
| 2026-11-04 (W8) | 博客终检删除 bench 时延数字；判定该组口径存疑、不外引。 |
| 2026-11-04 (W10) | 两份 AI 应用工程师简历清除 `9.18 / 8.70 / 3006→1107`，引用精确率 `0.651→0.766`。 |
| 2026-11-04 (W11) | 修正 `docs/bench-glm4flash.md`、`docs/token-curve.md` 的「可对外引用」误标；建立本真值表；公开四份脱敏证据。 |
