# 真实模型评测基线（v0.4-eval）

> 原始报告（不进仓库）：`reports/eval_20261016-015654.{json,md}`。
> 本文是该次实测的存档摘要。

- 时间：2026-10-16 01:56（GMT+8）
- 数据集：`config/eval_set.example.json`（10 条，rubric v1）
- 生成模型：`zhipu/glm-4-flash` ｜ 裁判模型：`zhipu/glm-4-air`
- 裁判输出可解析率：**100%**（rubric + JSON-only 输出格式有效）

| 维度 | 平均分 | 条数 |
| --- | --- | --- |
| 上下文一致性 | 5.0 | 1 |
| 事实可靠性 | 4.33 | 3 |
| 应拒答场景 | 5.0 | 2 |
| 追问合理性 | 3.5 | 2 |
| 面试风格符合度 | 4.5 | 2 |

## 诚实边界（引用前必读）

1. **自评偏差未消除**：裁判与生成同属智谱（不同型号），报告已自动打印 WARNING。
   彻底解法是多厂商对照（如 DeepSeek 当裁判），需要第二把 Key。
2. **无人工标注**：`human_score` 全空，kappa/QWK 一致性统计尚未启用；
   需抽 ≥30 条人工打分后再谈「裁判与人对齐」。
3. 样本 n=10，维度分数只能看相对短板（追问合理性 3.5 最低），
   不能当绝对质量结论。
4. 「追问合理性」恰好是全场最低分——与产品直觉一致（机械换题是面试机器人的
   常见病），是下一步 prompt 优化的首个靶点。

## 复现

```bash
AIMI_LLM__PROVIDER=zhipu AIMI_LLM__API_KEY=<key> \
AIMI_JUDGE__PROVIDER=zhipu AIMI_JUDGE__MODEL=glm-4-air AIMI_JUDGE__API_KEY=<key> \
  python -m src.cli eval --concurrency 3
```
