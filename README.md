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

## 七、下一步（按 6 周计划）

- **W1 Step1**：补一条 `/v1/chat` 非流式 vs `/v1/chat/stream` 的时延对比脚本，出 TTFT / P50 / P95 表 → 填进简历第一条
- **W2**：自建 RAG（移植 `medical-rag`：切分 → bge-small-zh → FAISS → BM25 → cross-encoder 重排）
- **W3**：手写 tool-call 循环做 Agent（题库检索 / 评分规则 / 追问策略）
- **W4**：100 条评测集 + LLM-as-judge + 人工校准 → 补上毕设「无法实时评估」的短板
- **W5**：上下文压缩 + Docker + 结构化日志
- **W6**：开源整理 + 技术博客 + 数字回填简历

每完成一步：打一次 git tag（`v0.1-service` / `v0.2-rag` / …），README 补一节，**立刻把真实数字填进简历**。
