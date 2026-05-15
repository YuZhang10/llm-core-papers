## HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face

### 决策卡片

- 年份：2023 / NeurIPS 2023
- 引用数：1,344，Semantic Scholar surface 快照
- Memory 层相关性：中-弱
- 是否值得先读：可作为 tool-use agent 背景读。memory 不是主贡献，但任务状态组织很有参考价值。

### 快速理解

HuggingGPT 把 ChatGPT 作为 controller，连接 Hugging Face 上的不同模型来完成多模态复杂任务。流程通常包括 task planning、model selection、subtask execution、response summarization。

从 memory layer 看，它的价值在于展示工具型 agent 的任务状态：用户请求被拆成子任务，子任务绑定工具或模型，执行结果再汇总给 controller。这里的 memory 主要是短期 orchestration state，而不是长期经验库。

### Memory layer 视角

- 记忆类型：任务计划、模型描述、子任务执行结果。
- 写入机制：每次工具/模型调用返回结果后加入任务上下文。
- 读取机制：controller 根据已有执行状态继续选择模型或总结答案。
- 价值：适合理解 tool execution trace 该如何进入上下文。
- 风险：模型选择错误或中间结果错误会传递到最终总结。

### 精读抓手

- task planning 的输出 schema。
- model selection 如何使用模型描述作为外部能力记忆。
- 多模态任务中 execution result 如何被压缩和汇总。
- 和 ReAct 对比：ReAct 更偏通用 thought/action loop，HuggingGPT 更偏模型工具编排。

### 链接

- arXiv: https://arxiv.org/abs/2303.17580
- Semantic Scholar author surface: https://www.semanticscholar.org/author/Yongliang-Shen/1471660296

