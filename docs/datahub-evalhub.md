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

### 3.5 门禁判定

三道可比性检查（不过就 `incomparable`，绝不硬比）：

1. `task` 相同；
2. `config_fingerprint` 相同；
3. `dataset_hash` 相同 —— **数据换了指标当然会变**，这是最常见的假警报。

比对逻辑：对共同数值指标算 `delta` 与 `delta_pct`，方向由指标名决定
（`latency_ms` / `avg_steps` / `model_calls` 等越低越好），
`abs(delta) >= 阈值` **或** `abs(delta_pct) >= 阈值` 即判退步。
两个条件并列是因为：baseline 很小时（如 0.04）相对跌幅会放大到失真，baseline 很大时
绝对差更直观。阈值默认 0.02（2 个百分点）。

### 3.6 为什么要逐条明细

指标退步 0.02 只是症状。要改代码得知道**是哪些 query 掉的**。所以 `eval_cases` 表存每条明细
（query / relevant / retrieved / rr / kind）。没有明细的评测只能证明「变差了」，
不能定位「哪里差」，等于没有归因能力。

---

## 4. API / CLI 速查

```
GET  /v1/data/datasets                     数据资产列表（含质量状态、被引用）
GET  /v1/data/datasets/{id}                详情（版本数、dirty、latest）
GET  /v1/data/datasets/{id}/quality        最新版本质量报告
POST /v1/data/sync[?force=true]            按依赖顺序全量同步
GET  /v1/data/lineage                      血缘
GET  /v1/data/snapshot/{id}                评测可用的数据快照

GET  /v1/eval/tasks                        可用任务
POST /v1/eval/runs                         {task, dataset_id, params, baseline_run}
GET  /v1/eval/runs[?task=&limit=&trusted_only=]
GET  /v1/eval/runs/{id}[/cases]            详情 / 逐条明细
POST /v1/eval/runs/{id}/gate?baseline=latest
```

```bash
python -m src.cli data ls
python -m src.cli data sync --dataset retrieval_eval
python -m src.cli data lineage
python -m src.cli evals run --task retrieval --baseline latest
python -m src.cli evals ls --task retrieval
python -m src.cli evals gate <run_id> --threshold 0.02
python scripts/smoke_platform.py            # 端到端演示，无需 Key
```

`baseline_run="latest"` 是实用约定：自动取同任务下最近一个 trusted run 作基线，
CI 里不需要人工记 run id。

---

## 5. 已知局限（面试会被追问，主动讲）

1. **会话状态仍在进程内 dict**，与中台的 SQLite 不是同一套持久化。中台解决的是「评测数据可追溯」，
   不是「会话可恢复」——后者是下一步。
2. **`agent` 任务在替身模式下 tool_call_rate 恒为 0**，这不是能力结论，是替身行为。
   所以它被标 `trusted=false`，且不能作基线。
3. **人工标注只覆盖 10 条中的 0 条**（`eval_set`），所以 kappa/QWK 目前算不出来。
   中台的 `human_labels_incomplete` warning 就是在提醒这件事。
4. **没有 CI 接线**：门禁能力已有，但还没在 GitHub Actions 里按 PR 触发。
5. **配置指纹不包含代码版本**：改了 `retriever.py` 的实现但配置不变时，指纹相同、会被判可比。
   这正是我们想要的行为（代码改动本来就该被门禁发现），但如果改动是「重构但等价」，
   也会产生一次「无变化」的 PASS——这是可接受的。

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
