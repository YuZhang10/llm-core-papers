---
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
paper_id: "arXiv:2503.14476"
authors: "Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, et al."
venue: "arXiv 2025"
date: "2026-05-21"
tags:
  - llm-rl
  - rlvr
  - dapo
  - grpo
  - long-cot
---

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

## 本地文件

- PDF: [DAPO_arXiv_2503.14476.pdf](DAPO_arXiv_2503.14476.pdf)
- arXiv source: [DAPO_arXiv_2503.14476_source.tar.gz](DAPO_arXiv_2503.14476_source.tar.gz)
- arXiv source dir: [DAPO_arXiv_2503.14476_source](DAPO_arXiv_2503.14476_source)
- arXiv: https://arxiv.org/abs/2503.14476
- Project: https://dapo-sia.github.io/

## 一句话理解

DAPO 的重点不是发明一个完全不同于 GRPO 的新范式，而是把 R1-style 长 CoT RLVR 里真正影响训练成败的工程细节公开出来：clip 怎么设、样本怎么筛、token loss 怎么算、超长回答怎么罚。

## 论文主线

DeepSeek-R1 证明了大规模 RL 可以显著激发 reasoning，但关键训练细节没有完全公开。DAPO 要解决的是可复现性问题：

```text
Qwen2.5-32B base
        -> math / reasoning prompts
        -> rollout multiple long-CoT responses
        -> rule-based verifier reward
        -> DAPO update
        -> AIME 2024 strong result
```

作者报告 DAPO 在 Qwen2.5-32B base 上达到 AIME 2024 50 分，并开源了基于 verl 的训练代码与处理数据。

## 方法直觉

### 1. Clip-Higher：给正向探索更多空间

PPO / GRPO 的 clip 是为了限制 policy update 太猛。但长 CoT RL 中，如果上界太保守，模型容易 entropy collapse，也就是很快收缩到少数模板化回答。

DAPO 把上下 clip 解耦：

```text
epsilon_low  控制负向/下界
epsilon_high 控制正向/上界
```

直觉是：坏方向仍然要压住，但好方向可以稍微放开一点，让模型保留探索能力。

### 2. Dynamic Sampling：不要训练没有梯度的组

GRPO 类方法依赖同一 prompt 下多个回答的 reward 差异。如果一组样本全对或全错，组内 advantage 接近没有有效学习信号。

DAPO 动态采样并过滤这类低信息组，把训练预算更多花在“有胜负差异”的 prompt 上。

### 3. Token-Level Policy Gradient Loss：长 CoT 不能粗糙平均

长回答里 token 数差异很大。如果按 response 粗糙平均，长 CoT 的梯度贡献可能被不合理缩放。DAPO 强调 token-level policy gradient loss，让长序列优化更平衡。

### 4. Overlong Reward Shaping：截断样本是噪声源

长 CoT RL 中，模型可能生成超过最大长度的回答。直接把截断回答判错，会把“可能快做完但被截断”的样本和“完全错误”的样本混在一起。

DAPO 引入 overlong filtering / soft overlong punishment，目标是减少截断带来的 reward noise。

## 流程图

```text
prompt q
  -> sample G long responses
  -> verifier gives rewards
  -> filter no-signal groups
  -> compute group-relative advantages
  -> apply decoupled clipping
  -> token-level policy gradient update
  -> handle overlong responses with shaping
```

## 关键实验

论文的 incremental table 很能说明 DAPO 的风格：每个技巧都不是单独制造奇迹，而是逐步把训练从“不稳但有潜力”推到“可用且可复现”。source 表格中可见：

| 逐步加入的组件 | AIME 2024 |
| --- | ---: |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-Level Policy Gradient Loss | 42 |
| + Dynamic Sampling / DAPO | 50 |

这些数字的意义在于：大规模 reasoning RL 的成败常常卡在训练细节，而不是只卡在“有没有 verifier”。

## 和 GRPO / R1 的关系

- GRPO 给了 critic-free group-relative policy optimization 的基本框架。
- R1 证明大规模 RLVR 能激发长思考和自检行为。
- DAPO 把这条线变成更可复现的开源 recipe。

可以把 DAPO 看成：

```text
GRPO core
  + long-CoT stability tricks
  + sampling efficiency tricks
  + overlong reward handling
  + open-source infrastructure
```

## 局限性

- 依然依赖可验证任务，开放式问答和 agent 任务不一定直接适用。
- DAPO 更偏工程 recipe，理论上并没有完全解释为什么这些技巧在所有规模上都稳。
- 动态采样可能改变训练分布，长期是否引入偏差需要额外评估。
- 论文很短，很多系统细节要结合代码和数据才能完全复现。

## 对我理解这条路线的意义

DAPO 的价值在于提醒我们：R1 后的 RLVR 不是一个干净的算法题，而是算法、采样、长度控制、batch 组织、inference/training 系统共同决定成败。它是“把 GRPO 真跑起来”的代表论文。

## 读这篇时抓住什么

抓住四个关键词：**Clip-Higher、Dynamic Sampling、Token-Level Loss、Overlong Shaping**。读完之后再看 GSPO / REINFORCE++，会更容易理解后续论文到底是在反驳 DAPO/GRPO 的哪个设计点。

