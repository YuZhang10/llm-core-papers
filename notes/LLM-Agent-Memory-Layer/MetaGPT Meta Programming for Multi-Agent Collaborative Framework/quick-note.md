## MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework

### 决策卡片

- 年份：2023
- 引用数：1,489，Semantic Scholar surface 快照
- Memory 层相关性：中
- 是否值得先读：可以。重点看 SOP 和中间产物如何变成团队协作记忆。

### 快速理解

MetaGPT 把软件工程中的标准流程编码进 multi-agent collaboration。不同角色的 agent 以 assembly line 方式协作，输出需求、设计、任务分解、代码等中间文档，用 SOP 约束流程，减少简单聊天链式调用带来的逻辑不一致。

memory 角度看，MetaGPT 的亮点不是个体长期记忆，而是团队级 artifact memory：每个阶段产物都成为后续 agent 的上下文和约束。对复杂工作流 agent，这种“文档即记忆”的模式比单纯聊天历史更稳定。

### Memory layer 视角

- 记忆类型：SOP、角色状态、中间文档、任务产物。
- 写入机制：每个 role agent 产出结构化 artifact。
- 读取机制：下游 agent 读取上游 artifact，而不是读取全部聊天噪声。
- 优点：降低多 agent 幻觉级联，提升流程可检查性。
- 风险：SOP 太硬会限制探索；文档质量差会污染后续步骤。

### 精读抓手

- SOP prompt sequence 如何定义。
- 角色之间交接的 artifact schema。
- 软件工程 benchmark 中 coherence 提升来自哪里。
- 对 memory 系统的启发：长期任务可以优先存“可交付物”，而不是全部对话。

### 链接

- arXiv: https://arxiv.org/abs/2308.00352
- GitHub: https://github.com/geekan/MetaGPT

