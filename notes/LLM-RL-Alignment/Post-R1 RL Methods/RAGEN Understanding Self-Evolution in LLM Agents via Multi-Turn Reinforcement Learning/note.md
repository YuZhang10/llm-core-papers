---
title: "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning"
paper_id: "arXiv:2504.20073"
authors: "Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, et al."
venue: "arXiv 2025"
date: "2026-05-21"
tags:
  - llm-rl
  - agent-rl
  - ragen
  - starpo
  - multi-turn
---

# RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning

## 本地文件

- PDF: [RAGEN_arXiv_2504.20073.pdf](RAGEN_arXiv_2504.20073.pdf)
- arXiv source: [RAGEN_arXiv_2504.20073_source.tar.gz](RAGEN_arXiv_2504.20073_source.tar.gz)
- arXiv source dir: [RAGEN_arXiv_2504.20073_source](RAGEN_arXiv_2504.20073_source)
- arXiv: https://arxiv.org/abs/2504.20073
- Code: https://github.com/RAGEN-AI/RAGEN

## 一句话理解

RAGEN 是 agent RL 的冷静版论文：它不只是提出 StarPO 训练框架，还系统展示多轮 LLM agent RL 会遇到 Echo Trap、reward variance collapse、浅层策略和幻觉式 reasoning 等问题。

## 论文主线

单轮数学 RLVR 的结构比较简单：

```text
prompt -> response -> final answer reward
```

agent RL 更像：

```text
state_0 -> thought/action_0 -> env feedback
state_1 -> thought/action_1 -> env feedback
...
trajectory -> cumulative reward
```

RAGEN 把这个过程形式化为多轮 MDP，并提出 StarPO：State-Thinking-Actions-Reward Policy Optimization，用 trajectory-level objective 优化整个交互轨迹。

## StarPO 方法直觉

StarPO 不是只看单个 response，而是把状态、推理、动作、环境反馈串成完整 trajectory：

```text
tau = {s_0, a_0, r_0, ..., s_K}
J(theta) = E_tau [R(tau)]
```

LLM 在每一轮生成 reasoning-guided actions，环境返回新状态和 reward。训练时对 trajectory-level reward 做 policy optimization。

## 实验环境

RAGEN 构造了几类环境：

| 环境 | 用来测试什么 |
| --- | --- |
| Bandit | 随机反馈和假设更新 |
| Sokoban | 多轮、不可逆动作、规划 |
| Frozen Lake | 多轮 + stochastic dynamics |
| WebShop | 更接近真实语言 grounding 和网页导航 |

这些环境比数学题更像 agent：行动会改变状态，错误可能不可逆，反馈有时随机。

## 关键发现

### 1. Echo Trap

训练可能突然进入 reward variance cliff 和 gradient spike。模型输出变得重复、策略变浅，像是只学会某个高频动作或口头理由，而不是学会真正规划。

### 2. Rollout design 很重要

初始状态多样性、交互粒度、采样频率都会影响训练。agent RL 不只是换个 reward 就能跑，它对 rollout 分布极其敏感。

### 3. 没有 reasoning-aware reward，reasoning 很难真正涌现

只给最终 task success reward，模型可能学会直接动作选择，或者生成看起来像 reasoning 但和环境状态不一致的幻觉思考。

### 4. SFT 仍然很强

论文附录里用 BFS 生成 trajectory 数据做 SFT，在某些任务上明显强于 rule-based RL。这个结果很诚实：agent RL 还没有简单碾压监督学习。

## 流程图

```text
environment state
    -> LLM thought/action
    -> environment transition
    -> reward
    -> next state
repeat K turns
    -> trajectory reward
    -> StarPO / StarPO-S update
```

## 和 R1 / GRPO 的关系

R1-style RLVR 是单轮或近似单轮 reasoning。RAGEN 关心的是多轮交互：

```text
R1 / GRPO:
  one prompt, one response, final verifier

RAGEN:
  stateful environment, repeated actions, trajectory reward
```

这意味着很多单轮 trick 不能直接搬。agent 里 reward 更稀疏、状态更多变、错误更难 credit assign。

## 局限性

- 环境仍偏 stylized，和真实 browser/code/data agent 还有距离。
- 使用的模型规模相对较小，扩展到更大模型可能出现不同现象。
- StarPO-S 是稳定化尝试，不是 agent RL 的最终答案。
- reasoning-aware reward 仍未真正解决，只是指出其必要性。

## 对我理解这条路线的意义

RAGEN 的价值在于给 agent RL 泼了一盆有用的冷水：多轮 RL 不会自然产生可靠 reasoning。没有细粒度 reward、稳定 rollout 和 trajectory credit assignment，模型可能只是学会“行动得像会推理”。

## 读这篇时抓住什么

抓住三个词：**Echo Trap、trajectory-level objective、reasoning-aware reward**。这篇更像问题地图，而不是最终算法。

