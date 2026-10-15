# AI 模拟面试官（ai-mock-interviewer）

把本科毕设《基于 VR 与语音交互的 AI 模拟面试官》从 C#/Unity 单体，重构成一套 **Python + FastAPI 的 LLM 应用工程**。

- 当前阶段：**Step 0 — 最小服务骨架**（对应《AI模拟面试官-改造执行手册》W1 前半段）
- 目标岗位：AI 应用工程师（校招 2026/2027）
- 许可证：MIT

> 口径红线（README 也会被面试官看）：**不写未实测的数字**、**不写「训练/微调模型」**（调的是云端 API）、**不把托管平台指标说成自己的**。

---

## 一、这一次（Step 0）做到什么程度

| 完成标准 | 状态 |
| --- | --- |
| `/v1/chat` 一次请求能拿到回答 | ✅ |
| `/v1/chat/stream` SSE 流式逐 token 吐出 | ✅ |
| `/health` 健康检查（含当前 provider / model） | ✅ |
| 配置集中在 `config/default.yaml`，密钥走环境变量 | ✅ |
| 统一的 LLM 客户端，可在 智谱 / 硅基流动 / DeepSeek / OpenAI 间切换 | ✅ |
| 时延埋点（TTFT / 总耗时 / token 数）落 JSONL | ✅（W1 Step1 用它出对比表） |
| 单元测试 + CI | ✅ |

还没做（后面几周）：自建 RAG、Agent / Function Calling、评测体系、记忆压缩、Docker。

---

## 二、目录结构

```
ai-mock-interviewer/
├── config/default.yaml         # 集中配置（provider / 模型 / 温度 / 埋点路径）
├── src/
│   ├── app.py                  # FastAPI 入口：/health、/v1/chat、/v1/chat/stream
│   ├── cli.py                  # 命令行：aimi serve / aimi chat
│   ├── metrics.py              # LatencyMeter（TTFT/总耗时）+ JSONL 落盘 + 分位数
│   ├── core/
│   │   ├── config.py           # YAML 默认值 + 环境变量覆盖（AIMI_ 前缀）
│   │   └── logger.py
│   ├── clients/
│   │   └── llm.py              # OpenAI 兼容客户端（手写 SSE 解析）+ 离线替身 EchoClient
│   ├── prompts/
│   │   └── interviewer.py      # 面试官角色 System Prompt（RAG / Agent 后续接这里）
│   └── schemas/
│       └── chat.py             # Pydantic 请求 / 响应模型
├── tests/
└── .github/workflows/ci.yml
```

---

## 三、快速开始（3 步）

### 1. 装依赖

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配 API Key（**没 Key 也能先把链路跑通**）

复制环境变量模板：

```bash
cp .env.example .env
```

二选一：

**A. 先不配 Key，用离线替身跑通链路**（推荐第一晚这样做）

```bash
export AIMI_LLM__PROVIDER=mock
```

**B. 配一个真模型**（智谱 GLM-4-Flash 免费）

```bash
export AIMI_LLM__PROVIDER=zhipu
export ZHIPU_API_KEY=你的key          # 或 export AIMI_LLM__API_KEY=你的key
```

### 3. 起服务

```bash
python -m src.cli serve            # 等价于 uvicorn src.app:app --reload
# 浏览器打开 http://127.0.0.1:8000/docs
```

或者不开服务，直接在终端里对话：

```bash
python -m src.cli chat
```

---

## 三之二、本地验证记录（tag `v0.1-service`）

在本机 Python 3.13 + Windows 上实跑通过，**没有使用任何 API Key**（全部跑在离线替身 `EchoClient` 上）：

```
pytest -q                         -> 32 passed
GET  /health                      -> {"provider":"mock","model":"echo-1","ready":false}
POST /v1/chat                     -> 200，带 trace_id 与 usage
POST /v1/chat/stream              -> 200 text/event-stream，28 个 SSE 帧，末帧含 ttft_ms=0.32
python -m src.cli chat            -> 终端多轮对话可用
logs/metrics.jsonl                -> 每次调用一行，含 ttft_ms / total_ms / token / turn
```

> 注意：上述延迟是**离线替身**的数据，只用于验证链路，**不能写进简历**。
> 真实时延要等你填了 Key 之后再跑一次才有意义。

## 三之三、跑时延基准（W1 Step1）

W1 的后半件事是「非流式 vs 流式」的时延对比表，**那张表是简历第一条两组数字的唯一来源**。
脚本已就位（`scripts/bench.py`），12 条应届 AI 应用工程师模拟面试问题在 `config/bench_questions.json`。

```bash
# 1) 起服务（用真实模型）
export AIMI_LLM__PROVIDER=zhipu && export ZHIPU_API_KEY=你的key
python -m src.cli serve

# 2) 另一个终端跑基准（≥30 次再看 P95 才有意义）
python scripts/bench.py --repeat 30 --concurrency 1 --tag glm4flash
```

产出 `reports/bench_<tag>_<ts>.md`（Markdown 表）与同名 `.json`（含每条原始记录）。样例：

| 链路 | 样本 | 客户端 TTFT P50 | TTFT P95 | 端到端 P50 | 端到端 P95 | 平均字数 |
| --- | --- | --- | --- | --- | --- | --- |
| 非流式 `/v1/chat` | n | — | — | … | … | … |
| 流式 `/v1/chat/stream` | n | … | … | … | … | … |

防呆设计：

- `health.ready=false`（离线替身）时，报告顶部自动打出 **MOCK 警示**——替身数字一律不得外引；
- `reports/` 已进 `.gitignore`，mock 产物不会误提交；
- 脚本自带 `--no-warmup` 开关：默认丢弃首次冷启动调用，避免把连接/模型预热算进 P95。

只验证脚本本身（不需要服务进程、不需要 Key）：

```bash
python scripts/bench.py --inline --repeat 2 --tag smoke
```

## 四、接口怎么用

**健康检查**

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","env":"dev","provider":"mock","model":"echo-1","ready":false,"version":"0.1.0"}
```

`ready=false` 表示当前跑在离线替身上，没有接真实模型。

**非流式**

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-1",
    "messages": [{"role":"user","content":"你好，我想应聘算法工程师"}],
    "profile": {"position":"AI 应用工程师","style":"专业严谨"}
  }'
```

返回：

```json
{
  "session_id": "demo-1",
  "reply": "你好。请先做一个自我介绍……",
  "model": "echo-1",
  "provider": "mock",
  "trace_id": "…",
  "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null},
  "latency_ms": 12.4
}
```

**流式（SSE）**

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"介绍一下你自己"}]}'
```

```
data: {"type":"delta","delta":"你好","trace_id":"…"}
data: {"type":"delta","delta":"，我","trace_id":"…"}
data: {"type":"done","ttft_ms":8.1,"total_ms":310.5,"trace_id":"…"}
data: [DONE]
```

---

## 五、厂商预设（`config/default.yaml` 里改 `provider` 即可）

| provider | base_url | 默认模型 | 读取的 Key 环境变量 |
| --- | --- | --- | --- |
| `mock` | — | `echo-1` | 不需要（离线替身） |
| `zhipu` | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash`（免费） | `ZHIPU_API_KEY` |
| `siliconflow` | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` | `SILICONFLOW_API_KEY` |
| `deepseek` | `https://api.deepseek.com` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` | `OPENAI_API_KEY` |

以上四家都是 **OpenAI 兼容的 `/chat/completions` 协议**，所以项目里只用一份 `OpenAICompatClient`。
评测时建议换一个模型当裁判（例如 DeepSeek 生成、GLM 打分），避免自评偏差。

环境变量覆盖规则：`AIMI_` 前缀 + 双下划线分层，例如 `AIMI_LLM__MODEL=deepseek-chat`。

---

## 六、埋点在哪里看

每次调用往 `logs/metrics.jsonl` 追加一行（可在配置里关掉）：

```json
{"ts":"2026-10-15T22:40:31","trace_id":"…","endpoint":"/v1/chat","provider":"zhipu",
 "model":"glm-4-flash","ttft_ms":412.3,"total_ms":1830.7,
 "prompt_tokens":256,"completion_tokens":188,"ok":true}
```

W1 Step1 拿 P50/P95 就用这个文件，`metrics.percentile()` 已经写好了。

---

## 六之二、评测体系（W4 地基，已就位）

这是**面试价值最高的一步**——论文第 39 页自陈「不能实时对面试结果进行分析」，这里正好把短板补上。

代码在 `src/eval/`，四个模块：

| 模块 | 职责 |
| --- | --- |
| `dataset.py` | 评测集契约：`EvalCase`（多轮对话 / 维度 / 参考要点 / 是否应拒答 / 人工分） |
| `judge.py` | LLM-as-judge：rubric 提示词 + JSON 输出解析（容忍代码块、前后废话） |
| `stats.py` | 严格一致率 / acc±1 / **Cohen's kappa** / **二次加权 kappa (QWK)** |
| `runner.py` | 跑批：生成 → 打分 → 一致性 → 出 Markdown 报告 |

跑一批：

```bash
python -m src.cli eval \
  --dataset config/eval_set.example.json \
  --generator zhipu --judge deepseek     # 裁判必须换成不同厂商
```

五个评测维度：**追问合理性 / 上下文一致性 / 事实可靠性 / 面试风格符合度 / 应拒答场景**。
样例 10 条在 `config/eval_set.example.json`，W4 的目标是把它填到 **100 条**。

三条已经写进代码的红线：

1. **裁判必须与生成不同模型**，同 provider 时报告自动打印自评偏差警告；
2. 评分是**有序等级**，所以主指标用 QWK（1 vs 5 的错远比 1 vs 2 严重），kappa 只作参考；
3. 小样本下 kappa 会失真（kappa paradox），报告同时给出 acc / acc±1 / QWK / 均值差，**不允许只报一个数**。

> 与 `medical-rag` 是同一套方法论（自建评测集 + 双盲对照 + 显著性检验），
> 但**用例与数据不许复用**——借的是方法，不是材料。

## 六之三、Agent：手写 tool-call 循环（W3 地基，已就位）

面试 Agent 和普通 chatbot 的分界线是**有状态 + 能调用工具**。代码在 `src/agent/`：

| 模块 | 职责 |
| --- | --- |
| `protocol.py` | 输出协议与解析：```tool 代码块 → `ToolCall`；宽松解析 + 坏格式回喂模型自纠 |
| `tools.py` | 工具注册表 + 参数校验 + 四个种子工具；错误信息刻意写成「可执行」的 |
| `state.py` | 会话状态：已覆盖维度、要点、轮次（进程内内存，接口按 Redis 预留） |
| `loop.py` | 手写 ReAct 循环：步数预算 / 死循环检测 / 工具异常不外溢 + `summarize_runs` |

四个种子工具：`search_question_bank`（检索，**W2 换向量检索时契约不变**）、
`get_interview_progress`、`save_candidate_note`、`finish_interview`。
题库 30 条在 `config/question_bank.json`。

```bash
python -m src.cli agent          # 终端对话，每步工具调用都会打印出来
GET  /v1/tools                   # 工具清单（含参数 JSON Schema）
POST /v1/chat/agent              # 带工具的对话，响应里带完整 trace
```

三个防御（每个真实 Agent 都得有，也顺带产出可统计字段）：

1. **步数预算**：`max_steps` 是允许的**工具次数**，产出自然语言终答的那一轮不占预算；
   超预算后追加一条「禁止再调工具」的指令再问一次（`stopped=max_steps_recovered`）
2. **死循环检测**：同名+同参数二次出现即警告，第三次强制收敛（`stopped=loop_guard`），
   且**不会**把模型的工具代码块原样吐给候选人
3. **工具异常不外溢**：未知工具 / 缺参 / 参数越界都转成 `ok=False` 的 observation 回喂模型自纠

可统计字段（由 `summarize_runs` 汇总，就是 W3 要往简历里填的那组数字）：
**工具调用成功率 / 完成率 / 平均模型调用次数 / 平均步数 / 停止原因分布**。

> ⚠️ 本机跑在离线替身上：进程一起、接口连通都已验证，但**工具是被脚本喂出来的，不是模型自己决定的**。
> 换成真实模型之前，成功率 / 完成率 / 步数这些数字只代表「机制生效」，**一律不得写进简历或任何对外材料**。

## 六之四、自建检索层（W2 地基，已就位）

`search_question_bank` 背后已经从「关键词打分」换成真正的检索层 `src/rag/`：

```
src/rag/
  tokenize.py   中文 bigram / 英文词级分词（不引 jieba）
  corpus.py     题库 -> Chunk 语料，字段加权靠重复词项放大 TF
  bm25.py       Okapi BM25，IDF 用非负变体（小语料专用）
  vector.py     向量检索 + 两个 embedder（见下）
  fusion.py     RRF 倒数排名融合
  retriever.py  Retriever 统一入口，三种策略
  evaluate.py   Recall@k / MRR@10 / Hit@1 + 区分度诊断
```

三种策略由 `config/default.yaml` 的 `rag.strategy` 决定，也可用环境变量覆盖（`AIMI_RAG__STRATEGY=hybrid`）：

| strategy | 依赖 | 说明 |
|---|---|---|
| `bm25` | 无 | **默认**。零依赖、可解释，小语料下的强基线 |
| `vector` | sentence-transformers | 稠密余弦，擅长字面不匹配 |
| `hybrid` | sentence-transformers | BM25 + 向量，RRF 融合 |

### 现在的重点：目前还没有真实语义嵌入

`sentence-transformers` 未安装时会自动退化为 **`HashingEmbedder`**（词袋 hashing 投影）。
它是**管道替身，不是语义模型**——「批处理」和「跑批」在它眼里毫无关系。

> ⚠️ **用替身跑出来的召回率、MRR 一律不得写进简历 / 面经 / 效果表。**
> `/health` 的 `semantic_retrieval` 字段会如实报告当前是不是替身（现在为 `false`），
> `scripts/eval_retrieval.py` 也会在报告顶部自动打印警示块。

装好真实模型后无需改代码：

```bash
pip install sentence-transformers          # 首次运行自动下载 bge-small-zh（约 90MB）
AIMI_RAG__STRATEGY=hybrid python -m src.cli serve
```

**换模型后第一件事是重新标定 `rag.min_score`**（默认 0.20）：不同模型的余弦分布差异极大，
照搬阈值会直接掐掉一部分正确召回——这点在替身身上已经验证过：
加上阈值后 `vector` 的 Recall@3 从 0.850 掉到 0.783，正是掐过头了。

### 评测怎么用

```bash
python scripts/eval_retrieval.py            # 三路对比，产出 reports/retrieval_*.md
python scripts/eval_retrieval.py --embed bge
```

评测集 `config/retrieval_eval.json` 是**手工标注的 30 条**，
其中 22 条 `literal`（字面改写）、8 条 `semantic`（语义改写，与原题几乎不共享词面）。
分两组统计，因为它们的结论完全不同：

| 分组 | BM25 MRR@10 | 含义 |
|---|---|---|
| `literal`（n=22） | ≈1.00 | 字面改写题，关键词方法就够 |
| `semantic`（n=8） | ≈0.44 | **语义改写题基本崩盘——这一组的差距才是换 Vector/Hybrid 的理由** |

> 语料只有 32 条，Recall@k 的绝对值没有参考价值；只看「同一份标注上哪个策略更好」。

### 这一层踩到的两个坑（都能在面试里讲）

1. **小语料的 IDF 会变负数**。教科书公式 `ln((N-df+0.5)/(df+0.5))` 在词出现超过半数文档时为负，
   于是「包含通用词」反而扣分。改用非负变体 `ln(1 + (N-df+0.5)/(df+0.5))`（Lucene 同款）。
2. **稠密检索没有「无命中」这个概念**。余弦相似度几乎不会等于 0，再不相干的文档也有 0.15 分，
   不设 `min_score` 阈值，Agent 就会被一堆垃圾召回带偏。BM25 天然有 >0 语义，不需要这步。

两者都写成了回归测试：`test_bm25_idf_is_non_negative_on_tiny_corpus`
与 `test_vector_retrieval_requires_similarity_threshold`。

## 七、下一步（按 6 周计划）

- **W1 Step1**：时延基准脚本已就位（`scripts/bench.py`），**等真实 Key 跑出 TTFT / P50 / P95** → 填进简历第一条
- **W2**：检索层已完成（本节）。剩下的活是装 `sentence-transformers` 用真实 bge-small-zh 重跑评测、标定 `min_score`、补 cross-encoder 重排
- **W3**：手写 tool-call 循环已就位，剩：扩充工具集 + 真实模型上的成功率 / 完成率实测
- **W4**：评测集从 10 条扩到 100 条 + 人工抽 ≥30 条校准 → 补上毕设「无法实时评估」的短板
- **W5**：上下文压缩 + Docker + 结构化日志
- **W6**：开源整理 + 技术博客 + 数字回填简历

每完成一步：打一次 git tag（`v0.1-service` / `v0.2-rag` / …），README 补一节，**立刻把真实数字填进简历**。
