---
title: "REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization"
paper_id: "arXiv:2501.03262"
authors: "Jian Hu, Jason Klein Liu, Haotian Xu, Wei Shen"
venue: "arXiv 2025"
date: "2026-05-21"
tags:
  - llm-rl
  - reinforce-plus-plus
  - critic-free
  - advantage-normalization
  - rlvr
---

# REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization

## 本地文件

- PDF: [REINFORCE_plus_plus_arXiv_2501.03262.pdf](REINFORCE_plus_plus_arXiv_2501.03262.pdf)
- arXiv source: [REINFORCE_plus_plus_arXiv_2501.03262_source.tar.gz](REINFORCE_plus_plus_arXiv_2501.03262_source.tar.gz)
- arXiv source dir: [REINFORCE_plus_plus_arXiv_2501.03262_source](REINFORCE_plus_plus_arXiv_2501.03262_source)
- arXiv: https://arxiv.org/abs/2501.03262

## 一句话理解

REINFORCE++ 认为 GRPO/RLOO 这类 critic-free 方法最大的问题不是没有 critic，而是 advantage normalization 太局部：只在一个 prompt 的小 group 内归一化会有偏且不稳；应该改成 global batch normalization。

## 论文主线

PPO 的 critic / value model 很贵，所以 R1 后很多方法走 critic-free 路线：

```text
sample multiple responses
-> compute rewards
-> subtract local baseline
-> normalize within prompt group
-> policy gradient update
```

GRPO 就是典型代表。但 REINFORCE++ 指出：prompt-level local normalization 通常 group size 很小，例如 4 或 8，样本统计量很噪；当组内 reward 接近时，std 很小，advantage 可能爆炸；而且这种估计在理论上有偏。

## 方法直觉

REINFORCE++ 的核心替换是：

```text
local normalization:
  A_i = (r_i - mean(group)) / std(group)

global normalization:
  A_i = (r_i - mean(global batch)) / std(global batch)
```

它的直觉很干净：全局 batch 通常远大于单个 prompt group，因此均值和方差估计更稳定。随着 batch size 变大，估计偏差会变小。

## 两个版本

### REINFORCE++

适合一般 RLHF，可以 k >= 1。它直接用 global batch 的统计量归一化 advantage，减少 critic-free 的计算和显存开销。

### REINFORCE++ w/ Baseline

适合 reasoning / agentic 场景，仍然采样同一 prompt 的多个回答。它先做 group mean subtraction，保留“同题比较”的 reshaping，再用 global statistics 做稳定归一化。

```text
reward
  -> subtract group mean
  -> normalize by global batch statistics
  -> policy update
```

这可以理解为：用 group mean 处理不同 prompt 的难度差异，用 global std 避免小 group 方差不稳。

## KL 处理

论文还讨论了 KL estimator 的选择，认为一些常见 KL loss 写法并不能真正约束 reference policy。它倾向使用能正确稳定估计 reverse KL gradient 的 k2 estimator。

这个细节说明：critic-free 不只是“去掉 value model”，KL、baseline、normalization 都要重新审视。

## 和 GRPO 的关系

| 维度 | GRPO | REINFORCE++ |
| --- | --- | --- |
| 是否 critic-free | 是 | 是 |
| baseline | prompt group mean/std | global batch mean/std 或 group mean + global std |
| group size 敏感性 | 高 | 较低 |
| 理论关注点 | group-relative reward | local normalization bias |
| 工程目标 | reasoning RL 简化 | 稳定、低显存、泛化更好 |

## 关键实验与观察

论文报告在一般 RLHF 和 complex agentic settings 中，REINFORCE++ 变体相比 GRPO / RLOO / PPO 有更稳定的 reward-KL tradeoff。小数据实验里，GRPO local norm 容易在训练集上过拟合，而 global normalization 泛化更好。

这里的启发是：local group 不一定是在学习“全局更好”，有时只是在学习“这个 prompt 内赢过同组样本”。

## 局限性

- Global normalization 依赖大 batch；小 batch 或 reward 分布很不均匀时也可能不稳。
- 对不同任务 reward scale 差异很大时，单纯 global statistics 可能需要额外校准。
- 论文版本较新，社区复现和工业采用还需要观察。
- 它主要修 advantage estimator，不直接解决 reward 稀疏或 step-level credit assignment。

## 对我理解这条路线的意义

REINFORCE++ 是理解 GRPO 局限的好补课材料。它告诉我们：critic-free policy optimization 的关键不是“去 critic”这一个动作，而是 baseline 和 advantage normalization 怎么估。

## 读这篇时抓住什么

抓住这句：**GRPO 的 local normalization 是一个小样本估计，可能把局部噪声放大成全局更新。** REINFORCE++ 的全部设计基本都围绕修这件事。

