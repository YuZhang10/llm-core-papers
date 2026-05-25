---
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
short_title: "DeepSeekMath / GRPO"
paper_id: "arXiv:2402.03300"
authors: "Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo"
venue: "arXiv 2024"
date: "2026-05-21"
tags:
  - llm-rl
  - grpo
  - rlvr
  - mathematical-reasoning
  - deepseek
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## 本地文件

- PDF: [DeepSeekMath_GRPO_arXiv_2402.03300.pdf](DeepSeekMath_GRPO_arXiv_2402.03300.pdf)
- arXiv source: [DeepSeekMath_GRPO_arXiv_2402.03300_source.tar.gz](DeepSeekMath_GRPO_arXiv_2402.03300_source.tar.gz)
- arXiv source dir: [DeepSeekMath_GRPO_arXiv_2402.03300_source](DeepSeekMath_GRPO_arXiv_2402.03300_source)
- arXiv: https://arxiv.org/abs/2402.03300
- citation snapshot: Pith page 显示 1,018 Pith papers citing it（2026-05-21 检索）

## 一句话理解

DeepSeekMath 有两层贡献：一是用大规模高质量数学语料继续预训练 DeepSeek-Coder-Base；二是提出 GRPO，用同一题目的多条采样回答做组内相对优势估计，省掉 PPO 的 value model，让数学 reasoning RL 更省显存、更容易扩展。

## 论文主线

数学 reasoning 的难点不是只缺 prompt，而是模型内部没有足够的数学语料、推理模式和可验证优化信号。作者的路线是：

```text
DeepSeek-Coder-Base
        -> math-related Common Crawl data selection
        -> 120B math tokens continued pretraining
        -> instruction tuning
        -> GRPO with math rewards
        -> DeepSeekMath
```

这篇被反复引用，主要是因为 GRPO 后来成为 DeepSeek-R1 和大量 open RLVR 工作的算法起点。

## GRPO 方法直觉

PPO 通常需要训练一个 value model / critic 来估计 baseline。对 LLM 来说，这很贵：你已经有 policy、reference、reward model，再加 critic，显存和系统复杂度都上来了。

GRPO 的想法是：同一个 prompt 采样一组回答，用组内 reward 的均值和方差作为 baseline，直接构造相对 advantage。

```text
prompt x
  -> sample y_1, y_2, ..., y_G from policy
  -> score each response with reward/verifier
  -> normalize rewards within the group
  -> use group-relative advantages to update policy
  -> keep KL-style constraint to reference policy
```

直觉上，如果同一题里某个回答比同组其他回答更好，它就应该被强化；如果更差，就被压低。这样不需要额外 value critic。

## 为什么数学适合这条路

数学题常常有标准答案，reward 可以比较清楚地自动验证。相比 open-ended assistant 偏好，数学 outcome reward 更便宜、更稳定。这就是 RLVR（reinforcement learning with verifiable rewards）后来突然变强的原因之一。

DeepSeekMath 的 RL 并不是凭空产生数学能力。前面的 continued pretraining 很重要：先把模型放到数学语料分布里，再用可验证 reward 做偏好强化。

## 关键实验

- DeepSeekMath 7B 在 MATH 上达到强结果，论文摘要中报告不使用外部工具和投票时为 51.7%。
- Self-consistency 64 samples 可以进一步提高到 60.9%。
- 作者把能力归因于两部分：数学数据选择 pipeline 和 GRPO 后训练。

## 和 DPO / PPO 的关系

| 方法 | 核心信号 | 是否 rollout | 是否 critic |
| --- | --- | --- | --- |
| PPO-RLHF | reward model 偏好分数 | 是 | 通常需要 |
| DPO | 离线 chosen/rejected pair | 否 | 不需要 |
| GRPO | 同题多样本的相对 reward | 是 | 不需要 |

DPO 更像离线偏好拟合，GRPO 更像保留在线探索的简化 PPO。reasoning RL 需要探索不同解法，所以 GRPO 这类 on-policy 方法更自然。

## 局限性

- GRPO 省掉 critic，但没有省掉 reward 设计；reward 一旦错误或可 hack，policy 仍会学偏。
- 组内归一化让训练依赖采样质量和 group size，rollout 成本仍然高。
- 数学题 reward 相对干净，迁移到开放任务、安全任务、长程 agent 任务会更复杂。
- DeepSeekMath 的主要提升来自数据 + RL 的组合，不能把所有效果都归因于 GRPO。

## 读这篇时抓住什么

这篇要抓住两个关键词：data first, RL second。没有数学 continued pretraining，GRPO 很可能只是优化一个没有足够候选解空间的模型。GRPO 的意义在于：当模型已经能产生多种候选推理时，用组内相对 reward 可以更便宜地把好推理路径放大。

