---
title: "Post-R1 RL Methods: DAPO, GSPO, REINFORCE++, VinePPO, PRM, Agent RL"
date: "2026-05-21"
tags:
  - llm-rl
  - rlvr
  - reasoning-rl
  - agent-rl
  - process-reward-model
---

# Post-R1 RL Methods: DAPO, GSPO, REINFORCE++, VinePPO, PRM, Agent RL

## 本地文件

| 方向 | Paper | arXiv | Note / 本地材料 |
| --- | --- | --- | --- |
| 大规模 RLVR recipe | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | 2503.14476 | [note](<DAPO An Open-Source LLM Reinforcement Learning System at Scale/note.md>) / [PDF](<DAPO An Open-Source LLM Reinforcement Learning System at Scale/DAPO_arXiv_2503.14476.pdf>) / [source](<DAPO An Open-Source LLM Reinforcement Learning System at Scale/DAPO_arXiv_2503.14476_source>) |
| Sequence-level RL | Group Sequence Policy Optimization | 2507.18071 | [note](<Group Sequence Policy Optimization/note.md>) / [PDF](<Group Sequence Policy Optimization/GSPO_arXiv_2507.18071.pdf>) / [source](<Group Sequence Policy Optimization/GSPO_arXiv_2507.18071_source>) |
| Critic-free RL | REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization | 2501.03262 | [note](<REINFORCE++ Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization/note.md>) / [PDF](<REINFORCE++ Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization/REINFORCE_plus_plus_arXiv_2501.03262.pdf>) / [source](<REINFORCE++ Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization/REINFORCE_plus_plus_arXiv_2501.03262_source>) |
| Credit assignment | VinePPO: Refining Credit Assignment in RL Training of LLMs | 2410.01679 | [note](<VinePPO Refining Credit Assignment in RL Training of LLMs/note.md>) / [PDF](<VinePPO Refining Credit Assignment in RL Training of LLMs/VinePPO_arXiv_2410.01679.pdf>) / [source](<VinePPO Refining Credit Assignment in RL Training of LLMs/VinePPO_arXiv_2410.01679_source>) |
| Process reward model | Let's Verify Step by Step | 2305.20050 | [note](<Lets Verify Step by Step/note.md>) / [PDF](<Lets Verify Step by Step/Lets_Verify_Step_by_Step_arXiv_2305.20050.pdf>) / [source](<Lets Verify Step by Step/Lets_Verify_Step_by_Step_arXiv_2305.20050_source>) |
| Process advantage verifier | Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning | 2410.08146 | [note](<Rewarding Progress Scaling Automated Process Verifiers for LLM Reasoning/note.md>) / [PDF](<Rewarding Progress Scaling Automated Process Verifiers for LLM Reasoning/Rewarding_Progress_arXiv_2410.08146.pdf>) / [source](<Rewarding Progress Scaling Automated Process Verifiers for LLM Reasoning/Rewarding_Progress_arXiv_2410.08146_source>) |
| Agent RL analysis | RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning | 2504.20073 | [note](<RAGEN Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning/note.md>) / [PDF](<RAGEN Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning/RAGEN_arXiv_2504.20073.pdf>) / [source](<RAGEN Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning/RAGEN_arXiv_2504.20073_source>) |
| Agent RL system | Agent Lightning: Train ANY AI Agents with Reinforcement Learning | 2508.03680 | [note](<Agent Lightning Train ANY AI Agents with Reinforcement Learning/note.md>) / [PDF](<Agent Lightning Train ANY AI Agents with Reinforcement Learning/Agent_Lightning_arXiv_2508.03680.pdf>) / [source](<Agent Lightning Train ANY AI Agents with Reinforcement Learning/Agent_Lightning_arXiv_2508.03680_source>) |

> 这组不是“破千高引主清单”，而是 DeepSeek-R1 之后 LLM RL / RLVR 快速演化的补课包。它们更适合回答：GRPO 之后，大家到底在修什么问题？

## 一句话理解

R1 之后的 LLM RL 研究可以粗略分成三条线：

```text
1. 怎么让 on-policy RLVR 更稳、更省、更可复现？
   -> DAPO, GSPO, REINFORCE++

2. 怎么把最终答案 reward 分配到中间推理步骤？
   -> VinePPO, PRM, PAV / Rewarding Progress

3. 怎么把单轮数学/代码 RL 扩展到多轮 agent？
   -> RAGEN, Agent Lightning
```

所以这组论文不是互相替代，而是在同一个系统的不同位置动刀：采样、优势估计、重要性比率、clip、process reward、trajectory decomposition、agent infrastructure。

## 放到一张图里

```text
prompt / environment state
        |
        v
current policy rollout
        |
        +--> single-turn reasoning response ------------------+
        |                                                     |
        +--> multi-turn agent trajectory                      |
                                                              v
reward source:
  - outcome verifier: final answer / unit tests / task success
  - process verifier: step-level progress / PRM / PAV
                                                              |
                                                              v
advantage / credit assignment:
  - group-relative: GRPO, DAPO
  - global-normalized: REINFORCE++
  - MC step value: VinePPO
  - process advantage: Rewarding Progress
  - transition-level: Agent Lightning
                                                              |
                                                              v
policy update:
  - token-level clipped objective: GRPO / DAPO
  - sequence-level clipped objective: GSPO
  - PPO-style with better credit: VinePPO
  - hierarchical / transition RL: Agent Lightning
```

## 1. DAPO: 复现 R1-style RL 的工程 recipe

DAPO 的影响力主要来自“把大规模 LLM RLVR 的配方公开”。它不是单个公式上的巨大理论跳跃，而是把一堆真正影响训练成败的细节组合起来，并在 Qwen2.5-32B base 上报告 AIME 2024 达到 50 分。

四个关键技巧：

| 技巧 | 解决的问题 |
| --- | --- |
| Clip-Higher | 解耦 clip 上下界，缓解 entropy collapse，保留探索 |
| Dynamic Sampling | 过滤掉没有学习信号的 prompt group，提高训练效率和稳定性 |
| Token-Level Policy Gradient Loss | 长 CoT 场景里按 token 组织 policy gradient，而不是粗糙地按 response 平均 |
| Overlong Reward Shaping | 处理被截断的超长回答，减少 reward noise |

读 DAPO 时要抓住一个点：**RLVR 的难点不只是 reward 是否正确，而是训练过程中大量样本其实没有梯度、或者梯度很脏。** DAPO 的价值在于把这些脏细节变成可复现 recipe。

## 2. GSPO: reward 是 sequence-level，更新也应该 sequence-level

GSPO 直接挑战 GRPO/DAPO 这类 token-level importance ratio 的设计。作者认为：LLM reasoning 的 reward 通常是整段 response 的 final outcome，那么 optimization unit 也应该和 reward unit 对齐。

GRPO 常见结构是每个 token 有自己的 importance ratio：

```text
pi_theta(y_t | x, y_<t) / pi_old(y_t | x, y_<t)
```

GSPO 改成整段 response 的 sequence likelihood ratio，并在 sequence level 做 clipping / rewarding / optimization：

```text
pi_theta(y | x) / pi_old(y | x)
```

直觉是：长序列中少数 token 的概率波动会把 token-level ratio 搅得很噪，尤其在长 CoT 和 MoE routing 场景更明显。GSPO 用整段序列作为重要性采样单位，声称比 GRPO 更稳定、更高效，也对 Qwen3 的 RL 改进有贡献。

如果说 DAPO 是“把 GRPO recipe 打磨好”，GSPO 更像是问：**GRPO 的基本优化单位是不是就选错了？**

## 3. REINFORCE++: critic-free 不等于一定要 prompt-level normalization

REINFORCE++ 站在 GRPO / RLOO 这条 critic-free 路线上，但指出一个容易被忽略的问题：很多方法按同一 prompt 的小 group 做 local advantage normalization，这在 group size 很小时会有偏、会不稳，也容易过拟合局部 prompt。

它的核心改动是：

```text
GRPO: normalize rewards within each prompt group
REINFORCE++: normalize advantages across the global batch
```

论文提供两个版本：

- REINFORCE++：适合一般 RLHF，甚至可以 k=1。
- REINFORCE++ w/ baseline：适合 reasoning / agentic 场景，先做 group mean subtraction，再用 global batch statistics 稳定归一化。

它的读法很简单：**如果 GRPO 的“组内相对优势”是一个小样本估计，那 REINFORCE++ 就是在问这个估计是不是太局部、太容易把噪声当信号。**

## 4. VinePPO: value critic 不好，不代表 credit assignment 不重要

很多 R1 后的算法都在去 critic：GRPO 没有 value model，DAPO 也沿着 critic-free recipe 做。但 VinePPO 提醒我们：去掉 critic 可能是因为 value network 在 LLM reasoning 上估不好，不代表 token / step-level credit assignment 不重要。

VinePPO 的核心是：用 Monte Carlo rollout 来估计某个中间 state 之后成功的概率变化，从而得到更准确的 step advantage。它保留 PPO 框架，但把 value network 替换成 MC-based value estimates。

重要结论：

- PPO 的 value network 在 reasoning-heavy tasks 中经常估不准中间状态价值。
- VinePPO 在 MATH / GSM8K 上比 PPO、GRPO、RLOO 等更好，论文报告可用更少 wall-clock time 达到更高峰值表现。
- 它的意义不是“又一个 PPO 变体”，而是把问题重新拉回 credit assignment：长推理里，不是所有 token/step 都该吃同样的 reward。

可以这样记：

```text
GRPO / DAPO: response-level reward -> group-relative advantage -> roughly same signal to whole response
VinePPO: response-level reward -> MC estimates at intermediate states -> step-level advantage
```

## 5. PRM: 从最终答案对错，到每一步是否靠谱

Let's Verify Step by Step 是 process reward model 路线的标志性论文。它比较 outcome supervision 和 process supervision：

- Outcome supervision：只看最终答案对不对。
- Process supervision：对每个中间步骤给反馈。

这篇释放 PRM800K：约 800K step-level human labels。论文报告 process-supervised model 在 MATH test subset 上达到 78.2%，并显示 active learning 让 process supervision 的数据效率提升约 2.6x。

PRM 的直觉非常清楚：如果最终答案错了，outcome reward 只告诉你“整段都不行”；process reward 能告诉你“从哪一步开始错”。这对 search、best-of-N、RL dense reward 都更友好。

但 PRM 的问题也明显：人工 step-level 标注贵，且“每一步是否正确”本身不总等价于“是否推动最终解题成功”。

## 6. Rewarding Progress / PAV: process reward 应该奖励 progress，而不是静态正确性

Rewarding Progress 对 PRM 做了一个很有价值的概念推进：process reward 不应该只是判断某一步是否“数学上正确”或“看起来合理”，更应该衡量这一步让未来成功概率增加了多少。

论文把这个叫做 progress / process advantage，并训练 Process Advantage Verifier (PAV) 去预测这种 step-level advantage。

核心观点：

```text
bad process reward:
  this step looks correct / relevant

better process reward:
  after taking this step, a prover policy is more likely to reach the correct answer
```

这解决了一个真实问题：很多步骤正确但没用，比如重述题目；有些步骤看起来平凡，但可能把搜索带到更容易解的位置。PAV 关心的是“状态改善”，不是单步审美。

论文报告 PAV 相比 ORM 在 test-time search 中有更高准确率和 1.5-5x compute efficiency；用于 online RL dense reward 时，也有 5-6x sample efficiency 和超过 6% accuracy gain 的结果。

## 7. RAGEN: agent RL 的冷水和清醒剂

RAGEN 很适合作为 agent RL 的现实提醒。它提出 StarPO，把 LLM agent 的多轮交互看作 trajectory-level policy optimization：

```text
state -> thinking -> action -> environment feedback -> next state
```

它在 Bandit、Sokoban、Frozen Lake、WebShop 等环境中研究 multi-turn RL，发现几个重要现象：

- Echo Trap：reward variance cliff + gradient spike，训练会突然塌到重复、浅层策略。
- rollout 设计很重要：初始状态多样性、交互粒度、采样频率都会影响训练。
- 仅有任务成功 reward 很难让真正 reasoning 涌现，模型可能学到 shallow strategies 或 hallucinated thoughts。
- 在部分环境上，SFT 仍明显强于纯 rule-based RL，说明 agent RL 还很早。

这篇的价值不是说“StarPO 已经解决 agent RL”，而是把 agent RL 的坑暴露得比较具体：**多轮环境里，reward 稀疏、状态随机、轨迹长、推理不可见，GRPO 那套单轮数学题经验不能直接搬过来。**

## 8. Agent Lightning: 把任意 agent 接进 RL 训练系统

Agent Lightning 更偏系统框架。它要解决的问题不是某个 benchmark，而是现实 agent 训练很难接 RL：

- agent 可能用 LangChain、OpenAI Agents SDK、AutoGen 或自研框架；
- 轨迹里混着 LLM 调用、工具调用、变量状态、控制流；
- 传统训练方法常要把整段轨迹拼成长序列再 mask，工程很脆。

Agent Lightning 的核心设计：

| 组件 | 作用 |
| --- | --- |
| Unified data interface | 把 agent execution 抽成 state/action/reward transitions |
| LightningRL | 对 episode-level return 做 credit assignment，再转成可训练 transition |
| Training-Agent Disaggregation | 训练服务和 agent runtime 解耦，agent 侧像调用 OpenAI API 一样接入 |
| AIR intermediate rewards | 从工具返回状态、系统监控信号里挖中间 reward，缓解稀疏奖励 |

它的战略价值在于：如果未来 agent RL 要落地，不能要求每个 agent 都重写成某个 RL 框架内部的 rollout loop。Agent Lightning 把 agent execution 和 training 解耦，是一个很现实的系统方向。

## 横向对比

| Paper | 最核心问题 | 一句话定位 | 当前影响力判断 |
| --- | --- | --- | --- |
| DAPO | 大规模 RLVR 如何复现和稳定训练 | GRPO-style recipe 工程化 | 高，开源 recipe 价值大 |
| GSPO | token-level ratio 是否噪声太大 | sequence-level policy optimization | 高，Qwen 系工业信号强 |
| REINFORCE++ | local normalization 是否有偏且不稳 | critic-free + global advantage normalization | 中高，算法补课重点 |
| VinePPO | 长推理 credit assignment 怎么做 | MC-based step value 替代 value network | 中高，洞察强但成本高 |
| Let's Verify Step by Step | 最终答案 supervision 是否太粗 | PRM800K + process supervision | 长期高，PRM 基础路线 |
| Rewarding Progress | process reward 到底该奖励什么 | reward progress / step-level advantage | 长期高，PRM -> PAV 的关键推进 |
| RAGEN | 多轮 agent RL 为什么难 | 轨迹级训练 + 失败模式分析 | 早期但重要，像问题地图 |
| Agent Lightning | 任意 agent 如何接入 RL | agent execution / training 解耦 | 早期但战略价值高 |

## 我会怎么排序阅读

如果你想最快建立框架：

1. DAPO：先看大规模 RLVR recipe。
2. GSPO：理解为什么 sequence-level 可能替代 token-level。
3. REINFORCE++：补 critic-free advantage estimation 的偏差和归一化问题。
4. Let's Verify Step by Step：理解 PRM 为什么比 ORM 更细。
5. Rewarding Progress：理解 process reward 为什么应该是 progress / advantage。
6. VinePPO：看 credit assignment 能不能不靠 learned critic。
7. RAGEN：看 agent RL 的失败模式。
8. Agent Lightning：看 agent RL 系统怎么接真实框架。

如果你只关心“和 GRPO 的关系”：

```text
GRPO
  -> DAPO: 保留 group-relative 思路，但加工程稳定 recipe
  -> GSPO: 改优化单位，从 token-level 到 sequence-level
  -> REINFORCE++: 改 advantage normalization，从 local 到 global
  -> VinePPO: 不满足 response-level advantage，重新做 step credit assignment
```

如果你只关心“agent RL”：

```text
RAGEN: 先告诉你多轮 agent RL 会怎么坏
Agent Lightning: 再告诉你怎么把真实 agent 轨迹喂给 RL 框架
PRM / PAV / VinePPO: 提供未来 agent credit assignment 的候选零件
```

## 读这组时抓住什么

这组文章共同说明一件事：R1-style RL 不是“有 verifier + 跑 GRPO”这么简单。真正决定训练能否放大的，是这些更细的工程和算法问题：

- 一个 prompt 下采样多少条，哪些样本有梯度，哪些只是浪费。
- reward 是整段级别，还是 step / transition 级别。
- advantage 是组内归一化、全局归一化、MC 估计，还是 learned verifier。
- clip 的单位是 token、sequence，还是 trajectory。
- 长 CoT 和 MoE routing 会不会让 token-level ratio 变成高噪声。
- 多轮 agent 的 reward 是否足够细，能不能分清“会行动”和“会编理由”。

我的当前判断：**DAPO / GSPO 是短期最该跟的 RLVR 工程主线；PRM / PAV / VinePPO 是长期 credit assignment 主线；RAGEN / Agent Lightning 是 agent RL 的早期地基。**
