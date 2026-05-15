## Reflexion: Language Agents with Verbal Reinforcement Learning

### 决策卡片

- 年份：2023 / NeurIPS 2023
- 引用数：Semantic Scholar surface 2,807；Princeton Pure 页面另列 Scopus 1,113
- Memory 层相关性：强
- 是否值得先读：非常值得。它是“agent 从失败经验中写入可复用文本记忆”的代表论文。

### 快速理解

Reflexion 的核心是不用更新模型权重，而是让 agent 根据任务反馈生成 verbal reflection，并把这些 reflection 存入 episodic memory buffer。下一次尝试同类任务时，agent 把这些记忆放回 prompt，引导策略改进。

这篇适合用来理解 agent memory 的一个关键分支：memory 不是事实知识库，而是自我经历、错误原因、策略提醒和行为约束。它把 trial-and-error learning 变成了 prompt-level 的经验积累。

### Memory layer 视角

- 记忆类型：episodic memory，主要保存自然语言反思。
- 写入触发：任务失败、环境反馈、外部或内部评价信号。
- 更新方式：追加式 buffer；不训练模型参数。
- 读取方式：下一轮决策时把 reflection 注入上下文。
- 价值：便宜、可解释、适合任务级自我改进。

### 精读抓手

- feedback signal 的来源：scalar reward、free-form feedback、self-generated feedback。
- memory buffer 内容到底怎么写，是否容易变成空泛提醒。
- HumanEval 91% pass@1 结果背后的 agent 配置。
- 对比 ReAct：ReAct 记住“发生了什么”，Reflexion 进一步记住“我学到了什么”。

### 链接

- arXiv: https://arxiv.org/abs/2303.11366
- Princeton Pure: https://collaborate.princeton.edu/en/publications/reflexion-language-agents-with-verbal-reinforcement-learning-2/

