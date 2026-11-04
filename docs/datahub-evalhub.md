# 数据中台与评测中台：设计说明

> 版本：v0.7 ｜ 对应代码：`src/datahub/`、`src/evalhub/` ｜ 测试：`tests/test_datahub.py`、`tests/test_evalhub.py`（共 40 个用例）

这份文档记录**为什么这样设计**，以及**哪些地方故意没做**。代码注释解释「这行在干什么」，这里解释「为什么需要这一层」。

---

## 1. 要解决的两个真实问题

### 1.1 数据散落，引用断裂不会报错

项目里有 4 份 JSON 数据，彼此不是独立的：

```
config/question_bank.json   80 条题库（被检索工具和检索评测同时消费）
config/retrieval_eval.json  60 条 query → relevant 指向题库 id   ← 引用 question_bank
config/eval_set.example.json 10 条对话评测用例
config/bench_questions.json  12 条时延基准脚本
```

2026-10 把题库从 30 条扩到 80 条时，「评测集里的 relevant 是否还都存在」只能靠人肉比对。
如果某条 id 被改掉，检索评测依然能跑完、依然出报告，只是那条 query 的召回永远拿不到——
**指标静默变低，没有任何地方报错**。这类错误最贵：它不崩，它只是让你得出错误结论。

### 1.2 评测是一次性脚本，无法比较

`scripts/eval_retrieval.py` 跑完写一份 md 就结束了。要回答「这次调参让 MRR 掉了几个点」，
只能翻历史 md 手工比——所以实际上不会有人真的去比，评测退化成「跑一次、看一眼、觉得还行」。

---

## 2. 数据中台

### 2.1 数据模型

| 概念 | 文件 | 说明 |
| --- | --- | --- |
| `DatasetSpec` | `models.py` | 登记信息：id / kind / path / tags / `depends_on` |
| `DatasetVersion` | `models.py` | 一个内容快照，主键是 `(dataset_id, content_hash)` |
| `QualityReport` | `models.py` | 按 hash 存档的质量报告（errors / warnings / stats） |
| `LineageEdge` | `models.py` | `from --depends_on--> to` |

存储：**SQLite**（`data/datahub.db`，`data/` 已在 `.gitignore`）。表就三张：
`datasets` / `dataset_versions` / `quality_reports`。

### 2.2 为什么版本号是 content_hash 而不是 v1/v2

人会忘记手动递增，而且两个分支可能都自称 v2。`content_hash = sha256(bytes)[:16]` 的含义是
**「同一 hash = 同一份字节」**，它不依赖任何人守规矩。这是「评测可复现」的地基：
run 记录里绑的是 hash，不是路径。路径会被悄悄改，hash 不会。

### 2.3 质量门禁的规则从哪来

每条规则都对应本项目真实踩过的坑，不是凭空凑的检查项：

| 规则 | 级别 | 对应后果 |
| --- | --- | --- |
| `dangling_reference`：relevant 指向不存在的 id | **error** | 召回永远拿不到那篇，指标静默变低 |
| `duplicate_query`：归一化后 query 重复 | **error** | 同一条被双倍计权，均值被少数样本绑架 |
| `duplicate_id`：题库/用例主键重复 | **error** | 检索结果 id 撞车，点开是另一道题 |
| `last_turn_not_user`：对话用例最后一句不是 user | **error** | 生成侧拿不到「当前提问」，评测无意义 |
| `dimension_gap`：某评测维度 0 条 | warning | 报告里那一维是空的，看起来却像「全测了」 |
| `human_labels_incomplete`：人工标注未覆盖 | warning | n<30 时 kappa/QWK 不可解读（kappa paradox） |
| `duplicate_question` / `bad_difficulty` | warning | 近似重复题、难度越界，检索语料质量下降 |

`duplicate_query` 用「去空白去标点 + 小写」的朴素归一化，**不用向量相似度**：
质检结论要能被人工复核，用黑盒判断会让人无法反驳也无法修。

### 2.4 三个语义（都不显然，但都必要）

1. **幂等**：内容没变 → `unchanged`，不产生新版本、不重跑校验。
   于是服务启动时可以无脑 `sync_all()`，不会每次启动都堆出一堆版本。
2. **blocked 不覆盖 latest**：新数据有 error 时**不写版本**，`latest` 仍指向上一个可用版本。
   否则「改坏了」会和「改好了」一样立刻生效。
3. **dirty 探测**：磁盘 hash ≠ 已登记最新 hash 时标 `dirty`。
   这是「评测结果对不上号」最常见的原因——文件改了但没同步。`snapshot()` 在 dirty 时返回
   `trusted_data=false`，评测中台据此拒绝把它当可信数据用。

### 2.5 故意没做的事

- **不搬迁数据**：JSON 仍是唯一事实源，数据库只存元数据。作品集项目里搞数据湖是过度设计。
- **不做数据格式转换层**：不做 Avro/Parquet/统一 schema registry。4 份 JSON 用不上，
  真上了只会增加「为了抽象而抽象」的维护成本。
- **不做权限**：单人项目，加 RBAC 是自娱自乐。

---

## 3. 评测中台

### 3.1 一次 run 的固定顺序

```
解析任务/数据集 → 绑定数据快照(hash) → 自动补同步 → 执行 → 合成 trusted → 算配置指纹 → 门禁 → 落库
```

顺序本身是设计：**先固定「用什么数据」，再跑**。反过来（先跑再补记数据）就会出现
「指标有、出处没有」，那批数字就永久不可解释了。

### 3.2 三类任务

| 任务 | 默认数据集 | 需要真实模型 | 指标 |
| --- | --- | --- | --- |
| `retrieval` | `retrieval_eval` | ❌ 零 LLM | Recall@k / MRR@10 / Hit@1，按时 literal/semantic 拆分 |
| `generation` | `eval_set` | ✅ | parse_rate / mean_score / 各维均分 / kappa / QWK |
| `agent` | `bench_questions` | ✅ | completion_rate / tool_call_rate / avg_tool_calls / avg_steps |

`retrieval` 能全离线跑，这点很重要——它让评测中台**在没有 API Key 的 CI 里也能被验证**，
40 个测试里有 6 个是端到端真跑检索的。

### 3.3 配置指纹：为什么必须存在

`config_fingerprint = sha256(影响结果的配置)`，包含 provider / model / strategy / embedder /
min_score / rerank / top_k 等。

没有它，最常见的误判是：把模型从 `glm-4-flash` 换成 `glm-4-air`，指标涨了 5 个点，
门禁报「改进」，而实际代码一行没改。**指纹不同就不许比**，这是让门禁可信的前提。

### 3.4 trusted：把红线编码进数据结构

项目有一条硬红线：**替身（EchoClient / HashingEmbedder）跑出的数字不得外引**。
过去这条只写在 README 里，靠人记。中台把它变成结构字段：

```python
record.trusted = bool(result.trusted and snap.get("trusted_data"))
#                    ↑ 任务自证                    ↑ 数据可信（已登记、质量过关、非 dirty）
```

判定规则（在 `tasks.py` 里由各任务自己回答，不是中台统一猜）：

| 场景 | trusted |
| --- | --- |
| BM25 检索 | ✅ 不涉及任何模型，数字天然真实 |
| vector/hybrid + 真实 bge 权重（`semantic=true`） | ✅ |
| vector/hybrid 退化到 HashingEmbedder | ❌ + warning |
| 生成/裁判任一跑在 EchoClient | ❌ + warning |
| Agent 跑在替身（不会发起工具调用） | ❌ + warning |
| 数据集 dirty / 质量 blocked | ❌ |

门禁**只允许 trusted run 之间比较**。于是「不许引替身数字」从自觉变成了系统约束：
拿替身分数当基线，门禁直接拒绝。

### 3.5 门禁判定（v0.8：三个条件同时满足）

三道可比性检查（不过就 `incomparable`，绝不硬比）：

1. `task` 相同；
2. `config_fingerprint` 相同；
3. `dataset_hash` 相同 —— **数据换了指标当然会变**，这是最常见的假警报。

判定退步要同时满足三条：

| # | 条件 | 作用 |
| --- | --- | --- |
| 1 | 方向变差 | `latency_ms` / `avg_steps` / `model_calls` 等越低越好，其余越高越好 |
| 2 | `abs(delta) >= 0.02` **且** `abs(delta_pct) >= 2%` | 绝对阈值防小 baseline 的相对放大，相对阈值防大 baseline 的绝对钝化 |
| 3 | Δ 的 95% CI 不跨 0（有明细时） | 区分"数字动了"和"统计上真的变差了" |

**v0.7 的写法是「绝对 or 相对」任一超阈即判退步，方向是错的。** 反例：baseline=0.04，
连掉 1 条 case 变成 0.036，相对跌幅 10% 直接触发报警——而小 baseline 恰恰是噪声最响的地方。
改成「且」之后，两个失真方向都被堵住。

### 3.6 显著性：为什么必须做、怎么做

60 条 query 的 Recall 标准误约 `sqrt(0.8*0.2/60) ≈ 0.05`。**0.02 的波动本来就在噪声里。**
只比点估计的门禁只有两种结局：阈值松到抓不住退步，或者紧到天天误报；误报几次之后
团队就会把它关掉——这比没有门禁更糟，因为它给了"我们有回归保护"的错觉。

做法（`src/evalhub/bootstrap.py`）：

1. **配对重采样**：两次 run 跑的是同一批 case，按 `case_id` 对齐后做差，
   把"这条 query 本身就难"的方差消掉，检出力显著高于两独立样本；
2. **还原校验**：明细里的字段名与汇总指标名不总是一一对应（`mrr@10` ← `rr`，
   `hit@1` ← `rr>=1`）。先验证"明细重建均值 ≈ 落库指标"（容差 0.02），
   对不上就**放弃检验**并写进 reasons——宁可说"这次判不了显著"，也不给假结论；
3. **不能重建的指标**（如 kappa/QWK，它们不是逐条均值）退回点估计，并在 reasons 里明说；
4. **判 pass 时也留痕**：把"跌了但 CI 跨 0"的指标逐条写进 reasons，
   因为使用者必须能区分"真的没退步"和"门禁没看见"。

实测（`scripts/smoke_platform.py`，全部可复现）：

| 场景 | 结论 | CI 表现 |
| --- | --- | --- |
| 同配置连跑两次 | `pass` | 9 个指标的 CI 全为 `[0, 0]` |
| 注入劣化 30%（每 10 条打坏 3 条） | `regressed` | 7 个指标 CI 整体为负，如 `mrr@10 [-0.383, -0.157]` |

第二行是刻意加的**反向验证**：门禁最大的风险不是误报，而是"从来没报过警，
没人知道它会不会报警"。注入数据不可外引，只用于证明判定逻辑有效。

### 3.7 为什么要逐条明细

指标退步 0.02 只是症状。要改代码得知道**是哪些 query 掉的**。所以 `eval_cases` 表存每条明细
（query / relevant / retrieved / rr / kind）。没有明细的评测只能证明「变差了」，
不能定位「哪里差」，等于没有归因能力。

v0.8 之后明细还多了一个用途：**它们是配对 bootstrap 的样本**。没有明细，门禁就只能比点估计。

指标退步 0.02 只是症状。要改代码得知道**是哪些 query 掉的**。所以 `eval_cases` 表存每条明细
（query / relevant / retrieved / rr / kind）。没有明细的评测只能证明「变差了」，
不能定位「哪里差」，等于没有归因能力。

---

## 4. API / CLI 速查

```
GET  /v1/data/datasets                     数据资产列表（含质量状态、被引用）
GET  /v1/data/datasets/{id}                详情（版本数、dirty、latest）
GET  /v1/data/datasets/{id}/quality        最新版本质量报告
POST /v1/data/sync[?force=true]            按依赖顺序全量同步          [写，需 Key]
GET  /v1/data/lineage                      血缘
GET  /v1/data/snapshot/{id}                评测可用的数据快照

GET  /v1/eval/tasks                        可用任务
POST /v1/eval/runs[?wait=true]             提交评测。默认 202 + run_id（异步）；    [写，需 Key]
                                           wait=true 阻塞到终态并返回 200
GET  /v1/eval/runs[?task=&limit=&trusted_only=]
GET  /v1/eval/runs/{id}[/cases]            详情 / 逐条明细
GET  /v1/eval/runs/{id}/progress           轻量进度（status/stage/progress）
POST /v1/eval/runs/{id}/gate?baseline=latest[&threshold=&abs_threshold=]         [写，需 Key]
POST /v1/eval/prune?keep_last=50&keep_trusted=20&dry_run=false                   [写，需 Key]
```

写接口带 `[写，需 Key]` 的，在 `auth.api_key` 非空时必须带 `X-API-Key` 头，
否则 401。读接口保持公开——观察不该需要钥匙。

### 异步 run 的完整用法

```bash
# 1. 提交，拿到 run_id（立刻返回，不等）
curl -s -X POST localhost:8000/v1/eval/runs \
  -H 'Content-Type: application/json' -H "X-API-Key: $KEY" \
  -d '{"task":"generation","params":{"limit":10}}'
# -> 202 {"id":"r-20261105-...","status":"pending","stage":"queued",...}

# 2. 轮询（stage 会走 queued -> binding_dataset -> running_task -> gating）
curl -s localhost:8000/v1/eval/runs/r-20261105-.../progress

# 3. 跑完看结论
curl -s localhost:8000/v1/eval/runs/r-20261105-...
```

```bash
python -m src.cli data ls
python -m src.cli data sync --dataset retrieval_eval
python -m src.cli data lineage
python -m src.cli evals run --task retrieval --baseline latest
python -m src.cli evals ls --task retrieval
python -m src.cli evals gate <run_id> --threshold 0.02 --abs-threshold 0.02
python -m src.cli evals prune --keep-last 50 --dry-run
python scripts/smoke_platform.py            # 端到端演示（含注入劣化对照），无需 Key
python scripts/check_docs_drift.py          # 文档数字与真实用例数是否一致
```

### 4.1 人工标注工作流（kappa/QWK 的前置条件）

裁判与人工的一致性是生成侧一切数字的可信度来源，而它需要 n≥30 的人工分。
工具链把「标注」从一句口号变成了三条命令：

```bash
python scripts/export_labeling_sheet.py                     # 1. 导出 CSV（Excel 可开，含 rubric 说明）
#    人工在 human_score 列填 1-5（评分标准见 docs/labeling/评分标准.md）
python scripts/import_human_labels.py docs/labeling/eval-set-标注表.csv
                                                            # 2. 校验并写回 eval_set（非法分/重复 id 直接拦截）
python -m src.cli data sync                                 # 3. 新版本登记 + 质量门禁
python -m src.cli evals run --task generation --baseline latest
                                                            # 4. 跑评测，报告直接给出 kappa / QWK
```

为什么是 CSV 而不是在线表格：`human_score` 是数据集的一部分，参与 content_hash
版本管理，CSV 是最小的可版本化载体。为什么是 30 条：n<30 时 kappa 会因
基础率不平衡而不可解读（kappa paradox），30 是最低可用样本量。


`baseline_run="latest"` 是实用约定：自动取同任务下最近一个 trusted run 作基线，
CI 里不需要人工记 run id。

---

## 5. 已知局限（面试会被追问，主动讲）

1. **会话状态仍在进程内 dict**，与中台的 SQLite 不是同一套持久化。中台解决的是「评测数据可追溯」，
   不是「会话可恢复」——后者是下一步。
2. **`agent` 任务在替身模式下 tool_call_rate 恒为 0**，这不是能力结论，是替身行为。
   所以它被标 `trusted=false`，且不能作基线。
3. **人工标注覆盖 30 条中的 0 条**（`eval_set` 已扩到 30 条，但分数还没标），
   所以 kappa/QWK 目前算不出来。中台的 `human_labels_incomplete` warning 就是在提醒这件事。
   标注工作流已就绪（见 4.1），标满 30 条后裁判一致性的数字就能第一次落地。
4. **配置指纹不包含代码版本**：改了 `retriever.py` 的实现但配置不变时，指纹相同、会被判可比。
   这正是我们想要的行为（代码改动本来就该被门禁发现）。`code_version` 单独记录但不参与判定，
   用途是事后归因——把两个 run 的 sha 一对比就知道该看哪几条 commit。
5. **后台任务是进程内的**：进程被 kill 时，正在跑的 run 停在 `running`。启动时
   `recover_stale_runs()` 会把残留的 pending/running 如实标为 failed（有测试守着），
   但**执行本身不会跨进程恢复**——要恢复必须上任务队列（Celery/RQ/arq），当前规模不值得。
6. **SQLite 单写者**：每次落库都是新连接 + commit，写并发靠 10s lock 兜底。单机评测没问题，
   多机部署时需要换成 Postgres —— 但那时真正该换的是整个持久层，不只是这张表。
7. **没有告警**：门禁判 `regressed` 只是落库，不会通知任何人。CI 化的下一步是
   "PR 里跑 retrieval 评测 + regressed 时阻断合并"，这需要先有 CI 里可复现的基线。
8. **回归门禁尚未按 PR 触发**：`scripts/check_docs_drift.py` 已接 CI，但评测门禁还没接——
   它的前置条件是"CI 里有稳定基线与固定评测集"，属于下一阶段。

---

## 6. 可能的追问与答法

**Q：为什么不直接用 MLflow / Weights & Biases？**
A：它们是更好的实验跟踪工具，但引入后本项目要维护一套额外服务与账号，面试官 clone 下来也跑不动。
这里的取舍是「clone 即跑」优先。真到团队场景，`runner.py` 的落库接口换成 MLflow Tracking 只是改一层。

**Q：SQLite 并发不会有问题吗？**
A：每次操作新建连接（`check_same_thread` 默认限制下最安全），写入频率极低（同步数据、跑评测各一次）。
并发量真上来，把 `store.py` 换成 SQLAlchemy + PG，上层不感知。

**Q：质量门禁会不会误拦正常数据？**
A：error 只卡「必填缺失 / 主键重复 / 引用断裂 / 用例结构不合法」这四类确定性问题；
有争议的（近似重复、维度缺口、标注未覆盖）一律只是 warning。宁可放过可疑的，不误杀可用的。

**Q：门禁判 pass 就代表没退步吗？**
A：只代表**共同指标**没超阈值退步。新增/删除的指标不参与比较（没有基线值），
指标之外的体感（生成质量、回答长度）它测不到。门禁是防线，不是质量保证。

**Q：为什么 `blocked` 不直接报错退出？**
A：一份数据坏了不该让整个服务起不来。返回 `blocked` + 质量报告，让其余数据集正常可用，
坏的显示为「没有可用版本」，`/health` 与 CLI 都能看到。
