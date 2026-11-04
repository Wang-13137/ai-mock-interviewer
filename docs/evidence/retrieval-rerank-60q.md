> 来源：本机 `reports/retrieval-rerank-60q_20261016-010128`，2026-10-16 实跑；本文件仅对模型权重的本机绝对路径做脱敏，数值与逐条明细未改动。
> 可引用范围见 [`docs/RESULTS.md`](../RESULTS.md)。

# 检索策略对比报告

| 策略 | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Hit@1 | 语义嵌入 |
|---|---|---|---|---|---|---|
| vector | 0.650 | 0.808 | 0.850 | 0.774 | 0.717 | bge-small-zh-v1.5 (local weights, not shipped with this repo) |
| hybrid | 0.642 | 0.825 | 0.867 | 0.767 | 0.700 | bge-small-zh-v1.5 (local weights, not shipped with this repo) |

评测集：60 条 query × 80 条文档。

## 按 query 类型拆分

| 策略 | literal MRR@10 | semantic MRR@10 |
|---|---|
| vector | 1.000 (n=30) | 0.547 (n=30) |
| hybrid | 1.000 (n=30) | 0.534 (n=30) |

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
| 系统提示词 写法 指令顺序 长度 | literal | llm-09 | 1.00 | 1.00 |
| token 成本估算 上下文占比 | literal | llm-16、llm-08 | 1.00 | 1.00 |
| asyncio 会话存储 加锁 并发 | literal | eng-10、llm-12 | 1.00 | 1.00 |
| P95 延迟升高 排查顺序 | literal | eng-13 | 1.00 | 1.00 |
| kappa paradox 类别不均衡 一致性 | literal | data-09 | 1.00 | 1.00 |
| 多重检验校正 BH Bonferroni 区别 | literal | data-01 | 1.00 | 1.00 |
| embedding 模型升级 向量索引重建 | literal | data-12 | 1.00 | 1.00 |
| 三年职业规划 为什么选这个岗位 | literal | beh-09 | 1.00 | 1.00 |
| 面试官让我证明某个改进真的有用，不能凭感觉 | semantic | proj-02、proj-13 | 0.50 | 1.00 |
| 怎么知道搜出来的内容是真相关而不是分数虚高 | semantic | proj-11 | 0.00 | 0.00 |
| 模型一本正经地胡说八道怎么办 | semantic | llm-06 | 0.00 | 0.00 |
| 对话轮数太多塞不下上下文了怎么办 | semantic | llm-05 | 1.00 | 1.00 |
| 机器人反复做同一个动作停不下来 | semantic | llm-04 | 0.00 | 0.25 |
| 页面上的回答一个字一个字往外蹦是怎么做到的 | semantic | llm-19、llm-03 | 0.00 | 0.00 |
| 同一句话换个说法就搜不到了怎么解决 | semantic | proj-12 | 1.00 | 0.33 |
| 付费接口突然不给用了系统怎么不崩 | semantic | llm-14 | 1.00 | 0.33 |
| 机器裁判和真人裁判打分不一致怎么量化差距 | semantic | llm-15、llm-07 | 0.50 | 1.00 |
| 上线一段时间后用户问的问题类型慢慢变了，怎么发现 | semantic | data-04 | 1.00 | 1.00 |
| 考题混进了练习题里，成绩就不算数了 | semantic | data-07、data-05 | 0.00 | 0.00 |
| 班里九十九个健康人一个病人，全猜健康也有九十九分 | semantic | data-02 | 1.00 | 0.33 |
| 对方服务器偶尔发脾气拒绝请求，客户端怎么办 | semantic | eng-03、eng-16 | 0.50 | 1.00 |
| 一个本地脚本想变成团队都能调的网络接口 | semantic | eng-01 | 1.00 | 1.00 |
| 几百个请求一起发出去怎么不把自己打挂 | semantic | eng-04 | 1.00 | 1.00 |
| 升级了翻译模型之后，之前存的指纹还有效吗 | semantic | data-12 | 1.00 | 0.33 |
| 别人改过的配置被我一保存就覆盖了 | semantic | eng-06 | 1.00 | 1.00 |
| 给领导汇报模型效果，光报数字被打回来了 | semantic | data-08 | 1.00 | 1.00 |
| 标注人手不够只能测几十条，这笔预算怎么花 | semantic | data-10 | 1.00 | 1.00 |
| 快到截止日期了发现最重要的指标不达标 | semantic | beh-10 | 1.00 | 1.00 |
| 上级坚持的方案我验证过是错的怎么办 | semantic | beh-12 | 1.00 | 1.00 |
| 工作里遇到完全没接触过的领域怎么上手 | semantic | beh-08 | 0.00 | 0.00 |
