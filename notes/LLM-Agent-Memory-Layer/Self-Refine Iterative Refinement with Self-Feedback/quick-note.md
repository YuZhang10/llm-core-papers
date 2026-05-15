## Self-Refine: Iterative Refinement with Self-Feedback

### 决策卡片

- 年份：2023 / NeurIPS 2023
- 引用数：2,992，Semantic Scholar surface 快照
- Memory 层相关性：中
- 是否值得先读：值得。它不是 autonomous agent 论文，但它是反思式短期 memory 的基础模式。

### 快速理解

Self-Refine 用同一个 LLM 反复执行 generate -> feedback -> refine。模型先生成初稿，再批评自己的输出，然后用反馈改写；整个过程不需要监督数据、训练或 RL。

从 memory layer 看，它把“上一版输出”和“反馈文本”变成短期可编辑 memory。很多 agent 的 reflection、critic、reviewer、debugger 模块都可以看成 Self-Refine 的系统化扩展。

### Memory layer 视角

- 记忆类型：当前任务内的 draft、feedback、revision trace。
- 写入机制：每轮自我反馈产生新的文本状态。
- 读取机制：下一轮 refinement 读取旧答案和 feedback。
- 生命周期：通常只在单个任务内有效，不做长期存储。
- 主要风险：没有外部反馈时，模型可能自信地循环修坏答案。

### 精读抓手

- feedback prompt 和 refine prompt 怎么写。
- 7 类任务上哪些任务真的受益，哪些任务提升有限。
- 为什么 GPT-4 也能被 test-time self-feedback 提升。
- 和 Reflexion 的区别：Self-Refine 改当前答案，Reflexion 把经验带到下一次任务。

### 链接

- arXiv: https://arxiv.org/abs/2303.17651
- Project: https://selfrefine.info/

