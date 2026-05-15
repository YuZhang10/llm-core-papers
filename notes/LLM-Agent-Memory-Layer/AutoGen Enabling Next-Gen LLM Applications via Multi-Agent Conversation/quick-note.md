## AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation

### 决策卡片

- 年份：2023
- 引用数：1,207，Semantic Scholar surface 快照
- Memory 层相关性：中
- 是否值得先读：可以放在核心 memory 三篇之后读。它更像 agent 框架，但对 conversation state 很重要。

### 快速理解

AutoGen 是一个多 agent 对话框架，允许开发者用多个可定制、可对话的 agent 组合 LLM、工具、人类输入和代码执行。它把复杂 LLM 应用抽象成 agent chat：不同 agent 通过消息历史和交互规则协作完成任务。

memory 视角下，AutoGen 的重点是“对话就是共享状态”。每个 agent 的上下文、历史消息、工具返回、人类干预，都构成任务 memory。它没有提出特别新的长期记忆算法，但推动了 conversation-centric agent architecture。

### Memory layer 视角

- 记忆类型：multi-agent conversation history。
- 写入机制：agent message、tool result、human input。
- 读取机制：参与对话的 agent 根据自身上下文和聊天历史生成下一步。
- 控制层：conversation pattern 决定哪些 memory 对哪些 agent 可见。
- 主要风险：历史过长、角色混淆、错误消息在多 agent 中传播。

### 精读抓手

- conversable agent 的抽象是否适合你的系统。
- group chat / nested chat / code execution 的状态边界。
- 哪些信息应该进入共享 history，哪些应该进入 agent private memory。
- 和 MetaGPT 对比：AutoGen 偏通用对话编排，MetaGPT 偏 SOP/文档流。

### 链接

- arXiv: https://arxiv.org/abs/2308.08155
- Semantic Scholar author surface: https://www.semanticscholar.org/author/Shaokun-Zhang/2116579935

