## Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents

### 决策卡片

- 年份：2022 / ICML 2022
- 引用数：1,482，Semantic Scholar surface 快照
- Memory 层相关性：弱-中
- 是否值得先读：如果你关心 embodied agent，可以读；若只关心长短期记忆机制，可排在后面。

### 快速理解

这篇研究 LLM 能否把自然语言高层任务转成可执行动作序列，例如把 "make breakfast" 分解并 grounding 到环境允许的动作。作者发现大模型可以生成中层计划，但 naive plan 往往不能直接映射到可执行动作，于是加入 demonstration conditioning 和语义翻译步骤。

memory 视角下，它不是 memory 论文，但它提醒 agent memory 不能只存文本，还要和环境 action space 对齐。对于 embodied agent，记忆需要服务可执行性：过去的 demo、当前场景、可行动作集合，都应该进入 planning context。

### Memory layer 视角

- 记忆类型：demonstration examples、当前可行动作、场景约束。
- 写入机制：本文主要不是持续写入，而是把已有 demonstration 作为上下文。
- 读取机制：LLM 从上下文中提取 actionable knowledge。
- 价值：说明 memory 与 action grounding 必须结合。
- 风险：语言计划合理但环境不可执行。

### 精读抓手

- high-level task 到 admissible action 的映射流程。
- VirtualHome 中 executability 与 correctness 的 trade-off。
- demonstration conditioning 对 plan grounding 的影响。
- 和 Voyager 对比：这篇偏 zero-shot planning，Voyager 偏 lifelong learning + skill memory。

### 链接

- arXiv: https://arxiv.org/abs/2201.07207
- Project: https://huangwl18.github.io/language-planner/

