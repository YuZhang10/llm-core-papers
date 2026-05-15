## Generative Agents: Interactive Simulacra of Human Behavior

### 决策卡片

- 年份：2023 / UIST 2023
- 引用数：1,280，Emergent Mind / Semantic Scholar surface 快照
- Memory 层相关性：强
- 是否值得先读：非常值得。这篇是 LLM agent memory architecture 的标志性论文之一。

### 快速理解

Generative Agents 让 25 个 NPC 在类似 The Sims 的沙盒小镇里生活、交流、计划和传播信息。论文最重要的部分不是仿真本身，而是 agent architecture：memory stream 记录所有自然语言经验，retrieval 按相关性、近期性、重要性取回记忆，reflection 把低层事件合成为高层洞察，planning 再利用这些记忆生成日程和行为。

如果你想先看 memory layer，这篇比很多通用 agent 框架更值得优先读，因为它直接回答了：记忆存什么、什么时候反思、如何检索、怎样影响计划。

### Memory layer 视角

- 记忆结构：memory stream，完整记录 perception / interaction / reflection。
- 检索信号：relevance、recency、importance。
- 反思机制：当重要性累积到阈值，触发高层 reflection 写回 memory stream。
- 下游使用：行为反应、长期计划、社交关系、事件传播。
- 主要风险：检索失败会导致遗忘；检索错误会导致不可信行为。

### 精读抓手

- Fig. 5 的 architecture：perceive -> retrieve -> reflect -> plan -> act。
- importance scoring 和 reflection trigger 的细节。
- ablation：observation、planning、reflection 对 believability 的贡献。
- 可以把它作为长期记忆系统的 baseline，然后再读 Reflexion 和 Voyager。

### 链接

- arXiv: https://arxiv.org/abs/2304.03442
- Emergent Mind: https://www.emergentmind.com/papers/2304.03442

