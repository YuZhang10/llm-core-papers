# LLM RL / Preference Alignment High-Citation Reading List

检索日期：2026-05-21（Asia/Shanghai）  
筛选口径：沿用当前 notes 里已有专题的口径，把“最近 4 年”按 2022-2026 处理；聚焦 LLM 后训练里的 RLHF、RLAIF、直接偏好优化、RLVR / reasoning RL。主清单要求公开引用快照超过 1000。引用数随 Semantic Scholar、Emergent Mind、Pith、Google Scholar 等数据库更新而变，这里保留当次检索快照和来源口径。

## 先读顺序

1. [Training language models to follow instructions with human feedback](<Training language models to follow instructions with human feedback/note.md>) - InstructGPT，把 SFT -> reward model -> PPO 这条现代 RLHF 管线产品化。
2. [Training a Helpful and Harmless Assistant with RLHF](<Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback/note.md>) - Anthropic HH-RLHF，把 assistant 的 helpful / harmless 偏好建模和在线迭代训练讲清楚。
3. [Constitutional AI: Harmlessness from AI Feedback](<Constitutional AI Harmlessness from AI Feedback/note.md>) - RLAIF 的代表，把人类逐条标注换成原则 + AI critique / AI preference。
4. [Direct Preference Optimization](<Direct Preference Optimization Your Language Model is Secretly a Reward Model/note.md>) - DPO，把 RLHF 的 reward model + PPO 改写成离线偏好分类目标。
5. [DeepSeekMath](<DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models/note.md>) - GRPO 的出处，把 PPO 的 critic 去掉，用 group-relative advantage 做数学 reasoning RL。
6. [DeepSeek-R1](<DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning/note.md>) - 作为核心延伸读。它对 2025 之后 reasoning RL 影响极大，但我这轮没有找到同样稳定的“破千引用”公开快照，所以不放进严格主清单。

## Post-R1 延伸

- [Post-R1 RL Methods](<Post-R1 RL Methods/note.md>) - DAPO、GSPO、REINFORCE++、VinePPO、PRM / PAV、RAGEN、Agent Lightning 的横向解读；适合作为 DeepSeek-R1 之后 RLVR / agent RL 的补课包。

## 主清单

| Paper | Year | Citation snapshot | 为什么是关键论文 |
| --- | ---: | --- | --- |
| Training language models to follow instructions with human feedback | 2022 | 9,967, [Emergent Mind / Semantic Scholar surface](https://www.emergentmind.com/papers/2203.02155) | InstructGPT；现代 LLM RLHF 三段式：SFT、reward model、PPO |
| Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback | 2022 | 1,816, [Emergent Mind / Semantic Scholar surface](https://www.emergentmind.com/papers/2204.05862)；搜索面另见 3,729 | Anthropic HH assistant；helpfulness / harmlessness、在线迭代偏好数据、KL 与 reward 的关系 |
| Constitutional AI: Harmlessness from AI Feedback | 2022 | 1,202, [Emergent Mind / Semantic Scholar surface](https://www.emergentmind.com/papers/2212.08073) | RLAIF；用 constitution + AI feedback 替代大量人工偏好标注 |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | 2023 | 7,658, [Semantic Scholar reference surface](https://www.semanticscholar.org/paper/Autoregressive-Direct-Preference-Optimization-Oi-Ukai/afd2968bd918ef9b09efcb152fc5f77179059202) | 直接偏好优化的分水岭；把 KL-regularized RLHF 改写为稳定的离线分类损失 |
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | 2024 | 1,018, [Pith papers citing it](https://pith.science/citations/c5006563-f3ec-438a-9e35-b7b484f34828) | GRPO 出处；critic-free PPO variant，后来 reasoning RL / RLVR 的基础算法之一 |

## 研究主线

### 1. RLHF 产品化：SFT + RM + PPO

InstructGPT 定义了很多人今天默认理解的 RLHF: 先用人工示范把模型拉到能回答指令，再用 pairwise ranking 训练 reward model，最后用 PPO 在 KL 约束下优化策略。它的意义不是某个损失函数，而是把“预训练语言模型”变成“可按人类偏好后训练的 assistant”。

HH-RLHF 把这个范式推到更完整的 assistant 设定：不只是有用，还要无害。它更像一篇系统工程论文，关心 preference model 如何扩展、线上迭代数据如何改善、helpfulness 和 harmlessness 是否冲突，以及 RL reward 被持续优化时会不会只是把模型推离原分布。

### 2. 反馈源扩展：从人类反馈到 AI feedback

Constitutional AI 的关键转向是：人类不用逐条写偏好，而是写原则。模型先按原则 critique / revise 自己的回答，再用 AI 对回答做偏好判断，最后进入 RL 阶段。这条线后来变成 RLAIF、自我反馈、自动红队、可扩展监督的一部分。

### 3. 直接偏好优化：绕开 reward model + PPO

DPO 的洞见是：在 KL-regularized RLHF 目标下，最优策略和 reward 存在闭式关系，所以可以直接用 policy log-prob ratio 表示隐式 reward。这样 reward model 和 PPO sampling loop 被压缩为一个离线 preference loss。它带来了工程稳定性，也引出了后续 IPO、KTO、ORPO、SimPO、CPO 等一整族方法。

### 4. Reasoning RL / RLVR：从偏好到可验证奖励

DeepSeekMath 的 GRPO 把 PPO 里的 value critic 去掉，改用同一 prompt 下多条 rollout 的 group-relative reward 做 advantage。数学题有可验证答案，天然适合 outcome reward。后来的 DeepSeek-R1 把这条线推到“base model 直接通过 RL 涌现长思考、反思、自检”的叙事中心。

## 未放入严格主清单但建议关注

- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 影响很大，我已放入专题目录作为延伸笔记；只是这轮没有找到稳定破千引用快照。
- SimPO: Simple Preference Optimization with a Reference-Free Reward - Semantic Scholar surface 约 876，接近破千，是 DPO 之后的强基线。
- KTO: Model Alignment as Prospect Theoretic Optimization - Semantic Scholar surface 约 911，接近破千，适合和 DPO / IPO 对照。
- RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment - Semantic Scholar surface 约 676，属于 ranking / reward-ranked SFT 路线。
- RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback - Semantic Scholar surface 约 254，重要但引用未破千。
- RLHF Workflow: From Reward Modeling to Online RLHF - 工程 recipe 很有用，但引用未破千。

## 读这个专题时抓住什么

- 奖励信号来自哪里：人工偏好、AI feedback、rule / verifier、reward model、implicit reward。
- 优化是 on-policy 还是 offline：PPO / GRPO 需要 rollout，DPO 类多半是离线 preference dataset。
- KL / reference policy 扮演什么角色：防止 reward hacking，也决定了“对齐强度”的上限。
- credit assignment 有多细：整段 outcome reward、token-level reward、process reward、group-relative reward。
- 失败模式是什么：reward model 过优化、长度偏置、偏好数据分布偏移、AI feedback 自举偏差、reasoning 格式奖励被 hack。
