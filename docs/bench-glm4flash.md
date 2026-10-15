# GLM-4-Flash 真实时延基线（v0.3-bench）

> 原始报告（不进仓库）：`reports/bench_glm4flash_20261016-015356.{json,md}`。
> 本文是该次实测的存档摘要，数字可对外引用（`ready=true`，provider=zhipu，非替身）。

- 时间：2026-10-16 01:05–01:53（GMT+8）
- 模型：智谱 `glm-4-flash`（免费档）
- 方法：inline ASGI 直连应用，12 条问题 × 重复 30 次 × 非流式/流式两条链路，并发 1
- 样本：720 次真实 API 调用，成功 720/720（两条链路各 360/360）

| 链路 | TTFT P50 | TTFT P95 | 端到端 P50 | 端到端 P95 | 平均字数 |
| --- | --- | --- | --- | --- | --- |
| 非流式 `/v1/chat` | — | — | 3252.9 ms | 9180.4 ms | 119.0 |
| 流式 `/v1/chat/stream` | 3362.8 ms | 8702.1 ms | 3363.0 ms | 8702.4 ms | 120.8 |

## 诚实结论

1. **短回答场景下 SSE 不降低 TTFT**（相对非流式端到端反而 -3.4%）。
   GLM-4-Flash 生成约 120 字的答案只需 3 秒出头，首 token 几乎就是全量；
   SSE 的价值在「逐字渲染的体感」，不在延迟数字。简历若引用，必须两条口径一起写。
2. TTFT 管体感、端到端管总成本，两个指标都保留。
3. P95 明显大于 P50（9.2s vs 3.3s），免费档存在长尾限流/排队，面试可讲
   「为什么生产系统要盯 P95 而不是均值」。
4. 若要改善 TTFT，方向是换更快的模型档位（glm-4-air / glm-z1-flash）或
   就近部署推理服务，而不是在传输层做文章——这由数据支撑，不是猜测。

## 复现

```bash
AIMI_LLM__PROVIDER=zhipu AIMI_LLM__API_KEY=<your-key> \
  python scripts/bench.py --inline --repeat 30 --tag glm4flash
```
