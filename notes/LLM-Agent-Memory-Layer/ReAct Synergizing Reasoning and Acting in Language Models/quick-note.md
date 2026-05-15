## ReAct: Synergizing Reasoning and Acting in Language Models

### 决策卡片

- 年份：2022 / ICLR 2023
- 引用数：6,455，Semantic Scholar 页面快照
- Memory 层相关性：中-强
- 是否值得先读：值得。它不是长期记忆论文，但它定义了许多 agent 系统的基本运行轨迹：thought/action/observation/history。

### 快速理解

ReAct 把 LLM 的推理 trace 和环境 action 交错起来，让模型一边想、一边查、一边更新计划。关键贡献不是“记忆库”，而是把 agent 的短期状态显式写进上下文：模型看到自己的 reasoning、已经采取的动作、外部环境或工具返回的 observation，再据此继续下一步。

这对 memory layer 的启发是：最小可用 memory 不一定先上向量库，很多 agent 的第一层 memory 就是结构化轨迹日志。后续的 Reflexion、工具 agent、浏览器 agent、代码 agent，都在这个轨迹日志上增加检索、压缩、反思或评分机制。

### Memory layer 视角

- 工作记忆：prompt 中的 thought/action/observation 序列。
- 写入机制：每一步 action 的结果被追加为 observation。
- 读取机制：LLM 在下一步生成时直接读取完整或截断轨迹。
- 主要风险：上下文增长、无关 observation 干扰、错误 trace 会被后续步骤继承。

### 精读抓手

- HotpotQA / FEVER 中，ReAct 如何用 Wikipedia API observation 降低 hallucination。
- ALFWorld / WebShop 中，action history 如何帮助更新计划。
- ReAct prompt 的格式可以作为 memory schema 的最小 baseline。
- 可以思考：如果轨迹太长，应该 summarization、retrieval，还是 selective replay？

### 链接

- arXiv: https://arxiv.org/abs/2210.03629
- Semantic Scholar: https://www.semanticscholar.org/paper/ReAct%3A-Synergizing-Reasoning-and-Acting-in-Language-Yao-Zhao/99832586d55f540f603637e458a292406a0ed75d

