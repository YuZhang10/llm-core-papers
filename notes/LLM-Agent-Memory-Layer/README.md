# LLM Agent Memory Layer High-Citation Reading List

检索日期：2026-05-12  
筛选口径：近 4 年内，也就是 2022-2026；优先选 LLM agent 或 agent-adjacent 论文；主清单要求公开引用来源显示引用数超过 1000。引用数会随数据库合并和更新时间变动，这里保留当次检索快照。

## 先读顺序

1. [[Generative Agents Interactive Simulacra of Human Behavior/quick-note|Generative Agents]] - 最直接的 memory stream / retrieval / reflection / planning 架构。
2. [[Reflexion Language Agents with Verbal Reinforcement Learning/quick-note|Reflexion]] - verbal reflection + episodic memory buffer，是 agent 自我改进 memory 的基本范式。
3. [[Voyager An Open-Ended Embodied Agent with Large Language Models/quick-note|Voyager]] - 把可执行 skill library 当成长期程序化记忆。
4. [[ReAct Synergizing Reasoning and Acting in Language Models/quick-note|ReAct]] - reasoning/action/observation 轨迹，是很多 agent 工作记忆接口的源头。
5. [[Self-Refine Iterative Refinement with Self-Feedback/quick-note|Self-Refine]] - 反思式反馈循环，可看成短期可编辑 memory。
6. [[Tree of Thoughts Deliberate Problem Solving with Large Language Models/quick-note|Tree of Thoughts]] - 把候选 reasoning state 外显成搜索树，适合理解 working memory/search memory。
7. [[AutoGen Enabling Next-Gen LLM Applications via Multi-Agent Conversation/quick-note|AutoGen]] - 多 agent 对话历史、工具执行与人类输入组成共享状态。
8. [[MetaGPT Meta Programming for Multi-Agent Collaborative Framework/quick-note|MetaGPT]] - SOP 和中间文档是团队级 procedural memory。
9. [[HuggingGPT Solving AI Tasks with ChatGPT and its Friends in Hugging Face/quick-note|HuggingGPT]] - 任务规划、模型选择、执行结果构成工具型 agent 的短期任务记忆。
10. [[Language Models as Zero-Shot Planners Extracting Actionable Knowledge for Embodied Agents/quick-note|Language Models as Zero-Shot Planners]] - embodied planning 早期高引基础，memory 相关性较弱但对环境状态和动作 grounding 有帮助。

## 引用数快照

| Paper | Year | Citation snapshot | Memory relevance |
| --- | --- | ---: | --- |
| ReAct | 2022 | 6,455, Semantic Scholar page | 中-强：轨迹式工作记忆 |
| Reflexion | 2023 | 2,807, Semantic Scholar surface；Princeton Pure 另列 Scopus 1,113 | 强：episodic reflection memory |
| Generative Agents | 2023 | 1,280, Emergent Mind / Semantic Scholar surface | 强：memory stream + reflection |
| Voyager | 2023 | 1,374, Semantic Scholar author/paper surface | 强：skill library as procedural memory |
| Self-Refine | 2023 | 2,992, Semantic Scholar surface | 中：self-feedback scratch memory |
| Tree of Thoughts | 2023 | 1,229, Emergent Mind / Semantic Scholar surface | 中：search-state memory |
| AutoGen | 2023 | 1,207, Semantic Scholar author/paper surface | 中：conversation state |
| MetaGPT | 2023 | 1,489, Semantic Scholar surface | 中：artifact/procedural memory |
| HuggingGPT | 2023 | 1,344, Semantic Scholar author/paper surface | 中-弱：tool execution state |
| Language Models as Zero-Shot Planners | 2022 | 1,482, Semantic Scholar surface | 弱-中：environment/action grounding |

## Memory 专项补充

- [[Mem0 Building Production-Ready AI Agents with Scalable Long-Term Memory/note|Mem0]] - 2025，agent 长期记忆的系统层方案：从对话中抽取 compact facts，做 ADD / UPDATE / DELETE / NOOP 维护，并用 `Mem0^g` 补 graph memory；不属于上面“高引主清单”的筛选口径，但很适合补齐 production memory layer 这一块。
- [[A-Mem Agentic Memory for LLM Agents/note|A-Mem]] - 2025，NeurIPS 2025，Zettelkasten 风格 agentic memory：把交互写成带 keywords / tags / context / links 的 atomic notes，并通过 link generation 和 memory evolution 让记忆网络持续自组织。
- [[LongMem Augmenting Language Models with Long-Term Memory/note|LongMem]] - 2023，模型侧长期 KV memory，适合理解 latent memory / memory bank 路线。
- [[Titans Learning to Memorize at Test Time/note|Titans]] - 2025，test-time neural memory，把长期记忆写进可在线更新的 memory module。
- [[MSA Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens/note|MSA]] - 2025，end-to-end latent memory scaling，把 routing 和 memory sparse attention 直接放进模型架构。

## 未放入主清单但建议后续看

- MemGPT: Towards LLMs as Operating Systems - memory layer 很核心，但我查到的公开引用快照没有稳定超过 1000；适合作为下一轮“memory 专项”加入。
- A Survey on Large Language Model based Autonomous Agents - memory module 综述很有用，但检索到的 Mendeley 快照约 862，未进主清单。
