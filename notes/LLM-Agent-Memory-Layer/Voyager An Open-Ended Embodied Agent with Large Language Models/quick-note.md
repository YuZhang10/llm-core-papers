## Voyager: An Open-Ended Embodied Agent with Large Language Models

### 决策卡片

- 年份：2023 / TMLR 2024
- 引用数：1,374，Semantic Scholar surface 快照
- Memory 层相关性：强
- 是否值得先读：非常值得。它展示了“可执行技能库”作为长期程序化记忆的 agent 形态。

### 快速理解

Voyager 是一个 Minecraft embodied lifelong learning agent。它有三个核心组件：automatic curriculum、ever-growing skill library、iterative prompting。memory 重点在 skill library：agent 把学到的复杂行为保存为可解释、可组合的代码技能，之后遇到新任务时检索并复用。

和 Generative Agents 的自然语言 memory stream 不同，Voyager 的长期记忆更接近 procedural memory：不是保存“我经历了什么”，而是保存“我会怎么做”。这对工具型 agent、代码 agent、机器人 agent 都很关键。

### Memory layer 视角

- 记忆类型：可执行代码技能库。
- 写入机制：任务尝试成功后，把行为封装为 skill。
- 读取机制：按任务上下文检索相关 skill，再组合执行。
- 反馈闭环：环境反馈、执行错误和 self-verification 进入 iterative prompting。
- 价值：减轻 catastrophic forgetting，让能力随经验复利增长。

### 精读抓手

- skill library 的表示、命名、检索和复用方式。
- automatic curriculum 如何驱动探索，不让 memory 只覆盖局部任务。
- error feedback 如何改写程序。
- 和 Reflexion 对比：Reflexion 存文字教训，Voyager 存可执行策略。

### 链接

- arXiv: https://arxiv.org/abs/2305.16291
- Project: https://voyager.minedojo.org/
- Semantic Scholar author surface: https://www.semanticscholar.org/author/Ajay-Mandlekar/49686756

