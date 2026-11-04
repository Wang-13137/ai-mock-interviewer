# AI 模拟面试官（ai-mock-interviewer）

把本科毕设《基于 VR 与语音交互的 AI 模拟面试官》从 C#/Unity 单体，重构成一套 **Python + FastAPI 的 LLM 应用工程**。

- 当前版本：**v0.7**（服务 / 评测体系 / Agent 双轨循环 / 自建 RAG / 记忆管理 / Docker / **数据中台 / 评测中台** 均已就位）
- 目标岗位：AI 应用工程师（校招 2027 届）
- 许可证：MIT

> 口径红线（README 也会被面试官看）：**不写未实测的数字**、**不写「训练/微调模型」**（调的是云端 API）、**不把托管平台指标说成自己的**、**替身（mock/哈希嵌入/抽取式摘要）跑出的任何数字不得外引**。
> **哪个数字能对外引用，以 [`docs/RESULTS.md`](docs/RESULTS.md) 为唯一真值表**；本页所有检索数字均可追溯到 [`docs/evidence/`](docs/evidence/) 的逐条明细。

---

## 一、全景：七个模块都做了什么

| 模块 | 状态 | 一句话 |
| --- | --- | --- |
| 服务骨架 `src/app.py` | ✅ | FastAPI + SSE 流式，统一 OpenAI 兼容客户端（智谱/硅基/DeepSeek/OpenAI 四家预设）+ 离线替身 |
| 时延基准 `scripts/bench.py` | ⚠️ 脚本就位，数字不外引 | 非流式 vs SSE 的 TTFT/P50/P95 对比脚本；历史产物口径存疑，**按 `docs/RESULTS.md` 一律不对外引用** |
| 评测体系 `src/eval/` | ✅ 地基就位 | LLM-as-judge + kappa/QWK 一致性 + 多指标强制同报 |
| Agent 循环 `src/agent/` | ✅ | **双轨协议**（native function calling + 手写 ```tool 兜底）+ 工具注册表 + 会话状态 + 三道防线 |
| 检索层 `src/rag/` | ✅ 默认 BM25；语义检索为**可选** | BM25 / 向量（bge-small-zh）/ RRF 混合三策略 + 标定过的阈值 + 实测后默认关闭的重排。**真实语义需另装 `requirements-semantic.txt` 并自备权重**；未装时会退化成可自报的替身 |
| 数据中台 `src/datahub/` | ✅ | 数据集登记 + content-hash 版本 + 质量门禁（含跨数据集**引用完整性**）+ 血缘 + 快照 |
| 评测中台 `src/evalhub/` | ✅ | 任务编排（检索/生成/Agent）+ 指标与逐条明细落库 + 配置指纹 + **回归门禁** + `trusted` 红线 |
| 记忆管理 `src/agent/memory.py` | ✅ | token 预算 + 滚动摘要，离线抽取式兜底 |
| Docker | ✅ | slim 镜像（不装 torch，容器内默认 BM25），非 root + 健康检查 |

测试：**200 passed**（pytest，全部离线替身，CI 免 Key）。

---

## 二、目录结构

```
ai-mock-interviewer/
├── config/
│   ├── default.yaml            # 集中配置（provider / rag / memory / 埋点路径）
│   ├── question_bank.json      # 80 条面试题库（五维度，难度 1-3）
│   ├── retrieval_eval.json     # 60 条手工标注检索评测集（literal 30 / semantic 30）
│   ├── bench_questions.json    # 12 条时延基准问题
│   └── eval_set.example.json   # 10 条生成质量评测种子
├── src/
│   ├── app.py                  # FastAPI：/health、/v1/chat*、/v1/tools、/v1/data/*、/v1/eval/*
│   ├── cli.py                  # python -m src.cli serve|chat|agent|eval|data|evals
│   ├── metrics.py              # LatencyMeter（TTFT/总耗时）+ JSONL 落盘 + 分位数
│   ├── core/config.py          # YAML 默认值 + AIMI_ 环境变量覆盖 + 厂商预设
│   ├── clients/llm.py          # OpenAI 兼容客户端（手写 SSE 解析）+ 离线替身 EchoClient
│   ├── prompts/interviewer.py  # 面试官 System Prompt
│   ├── agent/
│   │   ├── protocol.py         # 工具协议：手写 ```tool 解析（兜底）+ ToolSpec→OpenAI tools 转换
│   │   ├── tools.py            # 工具注册表 + 四个种子工具
│   │   ├── state.py            # 会话状态（asyncio 锁，接口按 Redis 预留）
│   │   ├── loop.py             # 手写 ReAct 循环 + native/handwritten 双轨提取 + 三道防线 + summarize_runs
│   │   └── memory.py           # 滚动摘要记忆（W5）
│   ├── rag/
│   │   ├── tokenize.py         # 中文 bigram / 英文词级（禁中文单字）
│   │   ├── bm25.py             # Okapi BM25，非负 IDF 变体（小语料专用）
│   │   ├── vector.py           # 向量检索 + HashingEmbedder 替身 + bge 真实嵌入
│   │   ├── fusion.py           # RRF 倒数排名融合
│   │   ├── rerank.py           # cross-encoder 精排（实测负收益，默认关闭）
│   │   ├── retriever.py        # 统一入口：bm25 / vector / hybrid
│   │   └── evaluate.py         # Recall@k / MRR@10 / Hit@1 + 饱和诊断
│   ├── eval/                   # LLM-as-judge 评测（dataset/judge/stats/runner）
│   ├── datahub/                # 数据中台
│   │   ├── models.py           # Dataset / Version / QualityReport / Lineage 契约
│   │   ├── store.py            # SQLite 元数据仓储（零新依赖）
│   │   ├── quality.py          # 分 kind 质量门禁 + 跨数据集引用完整性
│   │   ├── registry.py         # 同步编排：hash 幂等 / blocked 不覆盖 / dirty 探测 / 快照
│   │   └── api.py              # /v1/data/*
│   └── evalhub/                # 评测中台
│       ├── models.py           # Run / Gate / MetricDelta / TaskInfo 契约
│       ├── store.py            # SQLite：run 元数据 + 逐条明细
│       ├── tasks.py            # retrieval / generation / agent 三类任务
│       ├── gate.py             # 回归门禁：可比性检查 + 指标差集
│       ├── runner.py           # 编排：快照绑定 -> 执行 -> trusted -> 门禁 -> 落库
│       └── api.py              # /v1/eval/*
├── scripts/
│   ├── bench.py                # 时延基准（非流式 vs SSE）
│   ├── eval_retrieval.py       # 检索三策略对比，替身自动打警示
│   ├── calibrate_min_score.py  # 向量检索阈值标定
│   ├── smoke_platform.py       # 中台冒烟：同步 -> 跑评测 -> 与基线比较
│   └── publish_evidence.py     # 把 reports/ 脱敏后发布到 docs/evidence/
├── tests/                      # 200 个用例（全部离线替身，CI 免 Key）
├── docs/
│   ├── RESULTS.md              # 对外数字唯一真值表（可外引 / 禁止外引）
│   ├── datahub-evalhub.md      # 两个中台的设计取舍与面试问答
│   ├── evidence/               # 四份脱敏检索评测报告，含逐条明细
│   └── blog-draft.md           # 技术博客（已终检，待发布）
├── requirements-semantic.txt   # 可选：真实语义检索依赖
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## 三、快速开始

### 1. 装依赖

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # 零依赖最小集，默认 BM25

# 可选：复现 docs/evidence/ 里的真实语义检索数字
pip install -r requirements-semantic.txt              # 会连带安装 torch
# 并把 bge-small-zh-v1.5 权重放到 models/（约 95MB，不随仓库分发）
```

> **默认行为要说清楚**：不装可选依赖时 `vector` / `hybrid` 会自动退化成 `HashingEmbedder` 替身，
> `/health` 如实标记 `semantic_retrieval=false`，评测脚本会打印替身警告。
> 此时跑出的任何 Recall/MRR/Hit **不得对外引用**（见 `docs/RESULTS.md`）。

### 2. 配 API Key（**没 Key 也能把链路全跑通**）

```bash
# A. 离线替身（推荐第一晚这样做）
export AIMI_LLM__PROVIDER=mock

# B. 真模型（智谱 GLM-4-Flash 免费）
export AIMI_LLM__PROVIDER=zhipu
export ZHIPU_API_KEY=你的key
```

### 3. 起服务 / 终端对话

```bash
python -m src.cli serve      # http://127.0.0.1:8000/docs
python -m src.cli chat       # 纯对话
python -m src.cli agent      # 带工具的面试 Agent，每步调用可见
```

### Docker

```bash
docker build -t aimi .
docker run -p 8000:8000 -e ZHIPU_API_KEY=你的key -e AIMI_LLM__PROVIDER=zhipu aimi
```

镜像刻意不装 torch（容器内检索走 BM25）；需要语义检索时按
`requirements-semantic.txt` 自行构建镜像并挂载 `models/`。**镜像内默认不是真实语义检索**，
`/health` 会如实上报。

---

## 四、检索层的三个实测结论（都能当面试素材）

评测口径：80 条题库 × 60 条手工标注（30 literal / 30 semantic），真实 `bge-small-zh-v1.5` 本地权重。
**证据：[`docs/evidence/retrieval-bge-60q.md`](docs/evidence/retrieval-bge-60q.md) 及同名 `.json` 逐条明细，可逐格复算下表。**

| 策略 | Recall@3 | MRR@10 | Hit@1 |
| --- | --- | --- | --- |
| bm25 | 0.792 | 0.767 | 0.717 |
| vector | 0.800 | 0.814 | 0.767 |
| **hybrid** | **0.825** | **0.829** | **0.800** |

1. **结论会随评测集难度翻转。** 在简单的 30 条评测集（literal 为主）上纯向量反超混合；
   补了 22 条字面零重叠的 semantic 难例后，hybrid 反超——**评测集构成决定结论，先看分布再看数字**。
2. **两段式重排不是免费的午餐。** `bge-reranker-base` 在两代评测集上都是负收益：
   早期 30 条集 vector MRR **0.883→0.828**（[证据](docs/evidence/retrieval-bge-80.md) / [重排](docs/evidence/retrieval-rerank-80.md)）；
   最终 60 条集 vector **0.814→0.774**、hybrid **0.829→0.767**
   （[证据](docs/evidence/retrieval-bge-60q.md) / [重排](docs/evidence/retrieval-rerank-60q.md)）。
   短文本小语料上 bi-encoder 已接近天花板，重排加的是噪声。代码保留、默认关闭（`rag.rerank=none`）。
3. **阈值必须随模型和数据标定。** `min_score` 用 `scripts/calibrate_min_score.py` 扫描，
   平台区 0.30–0.40，代码与 `config/default.yaml` 统一取上沿 **0.40**（`DEFAULT_MIN_SCORE`）。
   换模型或扩语料后必须重跑；标定过程无独立归档报告，按 `docs/RESULTS.md` C 区只讲方法、不把扫描数值当硬证据。

> 语料只有 80 条，Recall@k 的绝对值没有参考价值；只看「同一份标注上哪个策略更好」。

### 这一层踩过的坑（都固化成了回归测试）

1. **小语料的 IDF 会变负数**：教科书公式在词出现超过半数文档时为负，「包含通用词」反而扣分。
   改用非负变体 `ln(1 + (N-df+0.5)/(df+0.5))`（Lucene 同款）。
2. **稠密检索没有「无命中」**：余弦几乎不为 0，不设 `min_score` 阈值，Agent 就被垃圾召回带偏。
3. **中文单字分词会假命中**：单字几乎命中所有条目，改 bigram（单字查询才退化 unigram）。

---

## 五、Agent 循环的三道防线

1. **步数预算**：`max_steps` 是允许的**工具次数**，终答轮不占预算；超预算追加「禁止再调工具」指令收敛；
2. **死循环检测**：同名同参二次警告、三次强制收敛，且**不会**把工具代码块原样吐给候选人；
3. **工具异常不外溢**：未知工具 / 缺参 / 越界都转成 `ok=False` 的 observation 回喂模型自纠。

每轮产出完整 trace（`POST /v1/chat/agent` 响应里的 `steps`），`summarize_runs` 汇总
工具成功率 / 完成率 / 平均步数 / 停止原因分布。**这些字段目前没有真实模型实测归档**，
按 `docs/RESULTS.md` B 区**不得对外引用具体数值**——机制可以讲，数字不能报。

---

## 六、记忆管理（W5）

长面试（20+ 轮）的上下文控制：历史估算 token 超预算（默认 3000）后，
最老的一批对话滚进摘要，只保留最近 6 条原文；摘要存在会话状态里**滚动更新**。
真实模型在线时用 LLM 写摘要，离线退化为抽取式（只保证链路可测，质量不外引）。
压缩发生时 trace 里会留一条 `kind=memory` 的步骤记录压缩前后 token 数。

---

## 七、评测体系红线

1. **裁判必须与生成不同模型**，同 provider 时报告自动打印自评偏差警告；
2. 评分是有序等级，主指标用 QWK，kappa 只作参考；
3. 小样本 kappa 会失真（kappa paradox），报告同时给 acc / acc±1 / kappa / QWK / 均值差，**不允许只报一个数**。

---

## 八、数据中台与评测中台（v0.8）

加这两层不是为了「听起来大气」，而是因为项目里真实存在两个**静默失败**：

1. **数据是散的**：题库、检索评测集、对话评测集、bench 脚本是 4 份独立 JSON，彼此有引用关系
   （评测集的 `relevant` 指向题库 id）。题库一改 id，评测集就指向空气——检索指标会悄悄变低，
   **不报错，但结论是假的**。
2. **评测是一次性的**：`scripts/eval_retrieval.py` 跑完写份 md 就结束了。想回答「这次改动让 MRR 掉了几个点」，
   只能人肉翻历史文件——所以实际上从来没有人真的去比。

### 数据中台 `src/datahub/`

做四件事，都不复杂，但都对应上面的问题：

| 能力 | 实现 | 解决什么 |
| --- | --- | --- |
| 登记 + 版本 | `content_hash`（sha256 前 16 位）作版本号 | 人会给版本号偷懒，hash 不会。同一 hash = 同一份字节，这是评测可复现的前提 |
| 质量门禁 | `quality.py` 分 kind 校验，error 直接拒绝入库 | 坏数据进不来，比每次消费时过滤便宜 |
| 引用完整性 | `retrieval_eval.relevant ⊆ question_bank.id` | 把「静默跑偏」变成 `dangling_reference` 硬错误 |
| 血缘 + 快照 | `depends_on` 声明依赖；评测绑定 hash 而非路径 | 路径会被悄悄改，hash 不会；`dirty` 标记「改了但没同步」 |

三个刻意设计：
- **blocked 不覆盖 latest**：数据改坏时不许顶掉上一个可用版本，否则「修一下就生效」会变成「改坏也生效」。
- **幂等**：内容没变就是 `unchanged`，不产生新版本、不重跑校验，服务启动可以无脑 `sync_all()`。
- **不搬迁数据**：JSON 仍是唯一事实源，SQLite 只存元数据。作品集项目里去搞数据湖是过度设计。

### 评测中台 `src/evalhub/`

把一次性脚本变成流水线：**任务注册 → 绑定数据快照 → 执行 → 落库（含逐条明细）→ 与基线比**。

三类任务：`retrieval`（Recall@k / MRR / Hit@1，零 LLM 可离线）、`generation`（LLM-as-judge 五维）、
`agent`（工具调用率 / 完成率 / 平均步数）。

**回归门禁**（`gate.py`）先做三道可比性检查，过不了就判 `incomparable`，绝不硬比：

| 检查 | 理由 |
| --- | --- |
| 任务相同 | 检索指标和生成分数没有可比性 |
| `config_fingerprint` 相同 | 换了模型/策略/阈值，涨跌不归因于代码改动 |
| `dataset_hash` 相同 | **数据换了指标当然会变**，这是最常见的假警报来源 |

**`trusted` 把红线编码进数据结构**——这是这层最有价值的部分：
每个 run 落库时判定「是否真实模型 + 真实嵌入产出」，替身产物一律 `trusted=false`，
而门禁**只允许 trusted run 之间比较**。于是「替身数字不得外引」从文档里的自觉，
变成了系统层面的硬约束：拿替身分数当基线，门禁会直接拒绝。

### 门禁不止比阈值，还比置信区间（v0.8）

点估计比阈值是不够的：60 条样本的 Recall 标准误约 0.05，**0.02 的波动本来就在噪声里**。
所以判退步要同时满足三条——方向变差、绝对差与相对差都超阈、**Δ 的 95% CI 不跨 0**：

```
$ python scripts/smoke_platform.py
门禁结论：pass（基线 r-...-c207，阈值 绝对 0.020 / 相对 2.0%）
  · 全部 9 个指标均未同时满足「方向变差 + 绝对差≥0.020 + 相对差≥2.0% + CI 显著」

  指标                         基线       本次        变化  95% CI                 方向
  mrr@10                 0.7672   0.7672     0.0%  [+0.0000, +0.0000]     越高越好
  recall@3               0.7917   0.7917     0.0%  [+0.0000, +0.0000]     越高越好
```

同一个脚本还会**注入劣化 30%** 做反向验证——门禁最大的风险不是误报，
而是"从来没报过警，没人知道它会不会报"：

```
门禁结论：regressed（阈值 绝对 0.020 / 相对 2.0%）
  · mrr@10 从 0.7672 变为 0.4972（-35.2%），95% CI [-0.3833, -0.1567] 整体偏离 0
```

双侧都验过，这个门禁才算数：**不误报**（相同 run 判 pass、CI 全 0）、**不漏报**（劣化即报警）。

### 平台化的五件"脏活"（v0.8）

设计漂亮不等于能上线。这四件是真实项目一定会撞上的：

| 问题 | 做法 |
| --- | --- |
| 长任务把 HTTP 拖死 | `POST /runs` 默认 **202 + run_id**，后台任务执行；状态机 `pending→running→succeeded/failed/rejected`，带 `stage`/`progress`，`run_timeout_s` 兜底防**幽灵 run** |
| 并发烧额度 | `max_concurrency` 信号量；`prune` 保留策略（最近 N 条 + 最近 K 个可信基线，在跑的不删） |
| 写接口裸奔 | `X-API-Key` 依赖（`hmac.compare_digest`），读接口公开；空 key 仅限本地 |
| 指标追不到代码 | 记录 `code_version`（git sha，脏工作区加 `-dirty`）+ `runtime` 快照。**刻意不进指纹**——否则门禁就抓不到代码改动 |
| 加字段就停服 | `PRAGMA table_info` + `ALTER TABLE` 轻量迁移，v0.7 老库直接升级可用（有测试守着） |

### 几条命令

```bash
python -m src.cli data ls                       # 数据资产 + 质量状态 + 谁引用谁
python -m src.cli evals run --task retrieval --baseline latest   # 跑评测并和上次比
python -m src.cli evals gate <run_id> --abs-threshold 0.02       # 单独复核门禁
python -m src.cli evals prune --keep-last 50 --dry-run           # 保留策略演练
python scripts/check_docs_drift.py              # 文档里的数字与实际是否一致（CI 会拦）
python scripts/smoke_platform.py                # 端到端演示：pass + 注入劣化 regressed
```

实测输出（BM25 默认配置，60 条标注 query，全离线）：

```
run r-20261105-001019-c207  status=succeeded  trusted=True
  code=022c884-dirty  dataset=retrieval_eval@<hash>
  mrr@10  0.7672   recall@3  0.7917   semantic_mrr@10  0.5511
```

> 注意这组是 **BM25** 的数字，和 `docs/RESULTS.md` 里 bge 语义检索的 0.825/0.829 不是一回事
> ——策略不同，不可混引。两组各自都能追溯到数据 hash 与 run 记录。

REST 侧：`GET /v1/data/datasets`、`POST /v1/eval/runs`、`GET /v1/eval/runs/{id}/cases`、
`POST /v1/eval/runs/{id}/gate`。`/health` 会如实报告 `datahub_enabled` / `evalhub_enabled`。

详细设计取舍与可能的追问见 [`docs/datahub-evalhub.md`](docs/datahub-evalhub.md)。

---

## 九、厂商预设

| provider | 默认模型 | Key 环境变量 |
| --- | --- | --- |
| `mock` | `echo-1`（离线替身） | 不需要 |
| `zhipu` | `glm-4-flash`（免费） | `ZHIPU_API_KEY` |
| `siliconflow` | `Qwen/Qwen2.5-7B-Instruct` | `SILICONFLOW_API_KEY` |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |

四家都是 OpenAI 兼容协议，共用一份 `OpenAICompatClient`。
环境变量覆盖：`AIMI_` 前缀 + 双下划线分层，如 `AIMI_RAG__STRATEGY=hybrid`。

### Agent 工具调用协议

`agent.protocol` 控制工具怎么交给模型，两条路径共用同一套执行 / 死循环检测 / 步数上限逻辑：

- `native`（默认）：把工具以 OpenAI `tools` 参数下发，由模型自行决定是否调用，格式由 API 保证合法；
- `handwritten`：把工具写成 ```tool 代码块塞进 system 提示词，自解析、可统计格式故障率，作为兼容免费额度模型与评测对照的兜底。

两条路径在 `AgentLoop._extract_call` 统一归一成内部 `ToolCall`。
环境变量可覆盖：`AIMI_AGENT__PROTOCOL=handwritten`。

---

## 十、已完成 / 下一步

**已完成**：推 GitHub 公开仓库、博客终检（`docs/blog-draft.md`）、四份脱敏检索证据公开、
对外数字真值表（`docs/RESULTS.md`）、版本号统一、语义检索可选依赖拆分、
**数据中台（登记/门禁/血缘/快照）与评测中台（编排/落库/回归门禁/`trusted` 红线）**。

**下一步（按优先级）**

1. 给 `scripts/smoke_native.py` 接一个 nightly/manual CI job，归档一份脱敏的真实模型工具调用输出
   —— 目前「v0.6 原生 function calling 生效」只有本机口述记录，**仓库里没有可验证证据**。
2. 在 CI 里跑 `evals run --task retrieval --baseline latest`，让每次 PR 自动出回归结论
   （中台已具备能力，缺的是流水线接线）。
3. 补统一错误契约（`{code,message,trace_id}`）、CORS / 限流 / 幂等、SSE 心跳与取消处理、429 指数退避。
4. 生成质量评测集 10 → 100 条，人工抽 30 条标完再谈裁判一致性（现在只有解析率，n=10）。
5. 会话状态从进程内 dict 落到 SQLite/Redis，结构化日志与 `trace_id` 注入。

**不会做的事**：把历史 bench 时延 / token 曲线数字「补测后回填」——那组口径已判定存疑、
一律不外引；若要一组可对外时延数据，必须另起一次有完整归档的实测并更新 `docs/RESULTS.md`。

发版纪律：改 `src/version.py` → 同步 `pyproject.toml` → 打同名 git tag →
README 与 `docs/RESULTS.md` 同步核对，**只把 A 区数字填进简历**。
