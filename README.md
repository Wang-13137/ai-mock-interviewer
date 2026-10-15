# AI 模拟面试官（ai-mock-interviewer）

把本科毕设《基于 VR 与语音交互的 AI 模拟面试官》从 C#/Unity 单体，重构成一套 **Python + FastAPI 的 LLM 应用工程**。

- 当前进度：**W1–W5 地基全部就位**（服务 / 时延基准 / 评测体系 / Agent 循环 / 自建 RAG / 记忆管理 / Docker）
- 目标岗位：AI 应用工程师（校招 2027 届）
- 许可证：MIT

> 口径红线（README 也会被面试官看）：**不写未实测的数字**、**不写「训练/微调模型」**（调的是云端 API）、**不把托管平台指标说成自己的**、**替身（mock/哈希嵌入/抽取式摘要）跑出的任何数字不得外引**。

---

## 一、全景：五个模块都做了什么

| 模块 | 状态 | 一句话 |
| --- | --- | --- |
| 服务骨架 `src/app.py` | ✅ | FastAPI + SSE 流式，统一 OpenAI 兼容客户端（智谱/硅基/DeepSeek/OpenAI 四家预设）+ 离线替身 |
| 时延基准 `scripts/bench.py` | ✅ 脚本就位 | 非流式 vs SSE 的 TTFT/P50/P95 对比；**等真实 Key 实测** |
| 评测体系 `src/eval/` | ✅ 地基就位 | LLM-as-judge + kappa/QWK 一致性 + 多指标强制同报 |
| Agent 循环 `src/agent/` | ✅ | 手写 tool-call 协议 + 工具注册表 + 会话状态 + 三道防线 |
| 检索层 `src/rag/` | ✅ **真实语义模型** | BM25 / 向量（bge-small-zh）/ RRF 混合三策略 + 标定过的阈值 + 实测后默认关闭的重排 |
| 记忆管理 `src/agent/memory.py` | ✅ | token 预算 + 滚动摘要，离线抽取式兜底 |
| Docker | ✅ | slim 镜像（不装 torch，容器内默认 BM25），非 root + 健康检查 |

测试：**135 passed**（pytest）。

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
│   ├── app.py                  # FastAPI：/health、/v1/chat、/v1/chat/stream、/v1/tools、/v1/chat/agent
│   ├── cli.py                  # python -m src.cli serve|chat|eval|agent
│   ├── metrics.py              # LatencyMeter（TTFT/总耗时）+ JSONL 落盘 + 分位数
│   ├── core/config.py          # YAML 默认值 + AIMI_ 环境变量覆盖 + 厂商预设
│   ├── clients/llm.py          # OpenAI 兼容客户端（手写 SSE 解析）+ 离线替身 EchoClient
│   ├── prompts/interviewer.py  # 面试官 System Prompt
│   ├── agent/
│   │   ├── protocol.py         # ```tool 代码块协议，宽松解析 + 坏格式回喂
│   │   ├── tools.py            # 工具注册表 + 四个种子工具
│   │   ├── state.py            # 会话状态（asyncio 锁，接口按 Redis 预留）
│   │   ├── loop.py             # 手写 ReAct 循环 + 三道防线 + summarize_runs
│   │   └── memory.py           # 滚动摘要记忆（W5）
│   ├── rag/
│   │   ├── tokenize.py         # 中文 bigram / 英文词级（禁中文单字）
│   │   ├── bm25.py             # Okapi BM25，非负 IDF 变体（小语料专用）
│   │   ├── vector.py           # 向量检索 + HashingEmbedder 替身 + bge 真实嵌入
│   │   ├── fusion.py           # RRF 倒数排名融合
│   │   ├── rerank.py           # cross-encoder 精排（实测负收益，默认关闭）
│   │   ├── retriever.py        # 统一入口：bm25 / vector / hybrid
│   │   └── evaluate.py         # Recall@k / MRR@10 / Hit@1 + 饱和诊断
│   └── eval/                   # LLM-as-judge 评测（dataset/judge/stats/runner）
├── scripts/
│   ├── bench.py                # 时延基准（非流式 vs SSE）
│   ├── eval_retrieval.py       # 检索三策略对比，替身自动打警示
│   └── calibrate_min_score.py  # 向量检索阈值标定
├── tests/                      # 135 个用例
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## 三、快速开始

### 1. 装依赖

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 需要语义检索再加：pip install sentence-transformers
```

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

镜像刻意不装 torch（容器内检索走 BM25）；需要语义检索时挂载 `models/` 并自行构建含 sentence-transformers 的镜像。

---

## 四、检索层的三个实测结论（都能当面试素材）

评测口径：80 条题库 × 60 条手工标注（30 literal / 30 semantic），真实 bge-small-zh-v1.5。

| 策略 | Recall@3 | MRR@10 | Hit@1 |
| --- | --- | --- | --- |
| bm25 | 0.792 | 0.767 | 0.717 |
| vector | 0.800 | 0.814 | 0.767 |
| **hybrid** | **0.825** | **0.829** | **0.800** |

1. **结论会随评测集难度翻转。** 在简单的 30 条评测集（literal 为主）上纯向量反超混合；
   补了 22 条字面零重叠的 semantic 难例后，hybrid 反超——**评测集构成决定结论，先看分布再看数字**。
2. **两段式重排不是免费的午餐。** bge-reranker-base 实测负收益（MRR 0.883→0.828），
   短文本小语料上 bi-encoder 已接近天花板，重排加的是噪声。代码保留、默认关闭（`rag.rerank=none`）。
3. **阈值必须随模型和数据标定。** `min_score` 用 `scripts/calibrate_min_score.py` 扫出来：
   0.30–0.40 是平台区，取上沿 0.40；0.60 时 semantic 子集 MRR 从 0.634 崩到 0.408。
   换模型或扩语料后必须重跑。

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
工具成功率 / 完成率 / 平均步数 / 停止原因分布——这组字段就是真实模型实测后要填的数字。

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

## 八、厂商预设

| provider | 默认模型 | Key 环境变量 |
| --- | --- | --- |
| `mock` | `echo-1`（离线替身） | 不需要 |
| `zhipu` | `glm-4-flash`（免费） | `ZHIPU_API_KEY` |
| `siliconflow` | `Qwen/Qwen2.5-7B-Instruct` | `SILICONFLOW_API_KEY` |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |

四家都是 OpenAI 兼容协议，共用一份 `OpenAICompatClient`。
环境变量覆盖：`AIMI_` 前缀 + 双下划线分层，如 `AIMI_RAG__STRATEGY=hybrid`。

---

## 九、下一步

- **真实 Key 实测**（等可用的智谱 Key）：`python scripts/bench.py --repeat 30 --tag glm4flash` → 出 TTFT/P95 → 打 `v0.3-bench` → 回填简历
- 真实模型上的 Agent 完成率 / 工具成功率实测（`v0.4-eval`）
- 生成质量评测集 10 → 100 条 + 人工抽 30 条校准裁判一致性
- 推 GitHub 远程 + 技术博客（草稿在 `docs/blog-draft.md`）

每完成一步：打 git tag、README 补一节、**只把真实测出的数字填进简历**。
