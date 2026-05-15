## Tree of Thoughts: Deliberate Problem Solving with Large Language Models

### 决策卡片

- 年份：2023 / NeurIPS 2023
- 引用数：1,229，Emergent Mind / Semantic Scholar surface 快照
- Memory 层相关性：中
- 是否值得先读：值得。它不是 agent memory 论文，但把 reasoning state 外显为可搜索结构，对工作记忆设计很有启发。

### 快速理解

Tree of Thoughts 把 Chain-of-Thought 从单条线扩展为搜索树。每个 thought 是一个中间状态，LLM 可以生成多个候选、评价状态、回溯，最后选择全局更好的路径。它本质上把 LLM 的临时思考过程外部化成一个可管理的数据结构。

对于 memory layer，这篇值得看的是“搜索状态如何存、如何评估、如何剪枝”。很多复杂 agent 的 planning memory 都需要维护候选计划、失败分支和评价分数，ToT 是很干净的起点。

### Memory layer 视角

- 记忆类型：多分支 reasoning/search state。
- 写入机制：生成新的 thought 节点。
- 读取机制：搜索算法选择当前最有希望的节点继续展开。
- 更新机制：value / vote 评分，必要时 backtracking。
- 价值：把不可见的推理过程变成可检查、可剪枝的工作记忆。

### 精读抓手

- thought decomposition：不同任务如何定义一个 thought。
- generation/evaluation/search 三个接口是否能泛化到 agent planning。
- Game of 24、creative writing、mini crossword 的差异。
- 和 ReAct 结合的可能：ReAct 是环境轨迹，ToT 是内部候选计划。

### 链接

- arXiv: https://arxiv.org/abs/2305.10601
- Emergent Mind: https://www.emergentmind.com/papers/2305.10601
- Code: https://github.com/princeton-nlp/tree-of-thought-llm

