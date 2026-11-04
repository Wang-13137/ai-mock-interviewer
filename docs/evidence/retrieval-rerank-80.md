> 来源：本机 `reports/retrieval-rerank-80_20261016-004236`，2026-10-16 实跑；本文件仅对模型权重的本机绝对路径做脱敏，数值与逐条明细未改动。
> 可引用范围见 [`docs/RESULTS.md`](../RESULTS.md)。

# 检索策略对比报告

| 策略 | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Hit@1 | 语义嵌入 |
|---|---|---|---|---|---|---|
| vector | 0.717 | 0.867 | 0.900 | 0.828 | 0.767 | bge-small-zh-v1.5 (local weights, not shipped with this repo) |

评测集：30 条 query × 80 条文档。

## 按 query 类型拆分

| 策略 | literal MRR@10 | semantic MRR@10 |
|---|---|
| vector | 1.000 (n=22) | 0.354 (n=8) |

`literal` = 字面改写（`...` 之外的同词替换），关键词方法通常能命中；
`semantic` = 语义改写，与原题几乎不共享词面——**这一组的差距才是换 Vector/Hybrid 的理由**。

## 逐条明细（第一个策略）

| query | 类型 | 相关 | Recall@3 | RR |
|---|---|---|---|---|
| RAG 重排 消融实验 | literal | proj-02、llm-01 | 0.50 | 1.00 |
| 知识库 chunk 怎么切分 重叠多少 | literal | proj-01 | 1.00 | 1.00 |
| 引用精确率 幻觉 降低编造 | literal | proj-03、llm-06 | 1.00 | 1.00 |
| McNemar 显著性检验 为什么不用准确率 | literal | proj-04、data-02 | 0.50 | 1.00 |
| 深度学习跑不过传统方法 负结果归因 | literal | proj-05 | 1.00 | 1.00 |
| 交叉验证 批次效应 数据泄漏 | literal | proj-06 | 1.00 | 1.00 |
| 质谱 峰 特征筛选 830 到 47 | literal | proj-07 | 1.00 | 1.00 |
| 可解释性 线性探针 遮挡法 置换检验 | literal | proj-08 | 1.00 | 1.00 |
| 自己搭一套 RAG 分几步 | literal | llm-01 | 1.00 | 1.00 |
| 向量库选型 FAISS Milvus PGVec | literal | llm-02 | 1.00 | 1.00 |
| 流式 TTFT 首字延迟 怎么优化 | literal | llm-03 | 1.00 | 1.00 |
| Agent 重复调用同一个工具 工程上怎么办 | literal | llm-04 | 1.00 | 1.00 |
| 多轮上下文太长 压缩 记忆 | literal | llm-05 | 1.00 | 1.00 |
| 大模型当裁判 打分 校准 一致性 | literal | llm-07 | 1.00 | 1.00 |
| Token 成本超标 从哪下手 | literal | llm-08 | 1.00 | 1.00 |
| 多重检验校正 FDR BH Bonferroni | literal | data-01 | 1.00 | 1.00 |
| 样本极度不平衡 准确率 AUC 不可靠 | literal | data-02 | 1.00 | 1.00 |
| 上线后数据分布漂移 无标注怎么发现 | literal | data-04 | 1.00 | 1.00 |
| 上游接口超时 429 重试 降级 | literal | eng-03 | 1.00 | 1.00 |
| 并发跑几百条请求 怎么控制并发度 异步一定更快吗 | literal | eng-04 | 1.00 | 1.00 |
| 脚本改成 HTTP 服务 接口和错误处理怎么设计 | literal | eng-01 | 1.00 | 1.00 |
| 为什么从生物医学工程转做 AI | literal | beh-02 | 1.00 | 1.00 |
| 模型评分和人工评分差得远，怎么处理 | semantic | llm-07 | 1.00 | 0.50 |
| 用户抱怨回答半天才出来，从哪查起 | semantic | llm-03 | 0.00 | 0.00 |
| 换一批样本结果就崩了，问题出在哪 | semantic | proj-06 | 0.00 | 0.00 |
| 能不能先拿全量数据算好标准化参数再切分 | semantic | data-05 | 1.00 | 1.00 |
| 机器人反复执行同一个动作停不下来 | semantic | llm-04 | 1.00 | 0.50 |
| 聊到后面就把开头说过的内容忘了 | semantic | llm-05 | 0.00 | 0.00 |
| 月底账单超预算，从哪几个地方省 | semantic | llm-08 | 1.00 | 0.50 |
| 挑了很多变量但模型好像把答案背下来了 | semantic | data-03 | 1.00 | 0.33 |
