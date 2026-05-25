---
title: "VinePPO: Refining Credit Assignment in RL Training of LLMs"
paper_id: "arXiv:2410.01679"
authors: "Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, Nicolas Le Roux"
venue: "ICML 2025"
date: "2026-05-21"
tags:
  - llm-rl
  - vineppo
  - credit-assignment
  - ppo
  - monte-carlo
---

# VinePPO: Refining Credit Assignment in RL Training of LLMs

## 本地文件

- PDF: [VinePPO_arXiv_2410.01679.pdf](VinePPO_arXiv_2410.01679.pdf)
- arXiv source: [VinePPO_arXiv_2410.01679_source.tar.gz](VinePPO_arXiv_2410.01679_source.tar.gz)
- arXiv source dir: [VinePPO_arXiv_2410.01679_source](VinePPO_arXiv_2410.01679_source)
- arXiv: https://arxiv.org/abs/2410.01679
- Code: https://github.com/McGill-NLP/VinePPO

## 一句话理解

VinePPO 的核心观点是：PPO 的 value network 在 LLM 长推理里经常估不准中间步骤价值，但这不代表 credit assignment 不重要。VinePPO 用 Monte Carlo rollout 估计中间 state 的成功概率变化，替代 learned value critic。

## 论文主线

LLM reasoning 的 reward 往往只在最后给：

```text
long reasoning trace -> final answer -> correct / incorrect
```

问题是：如果答案错了，到底是哪一步害的？如果答案对了，哪些步骤真正有贡献？

GRPO / RLOO / rejection finetuning 常常把整段 response 的 reward 平均施加到所有 token 上。PPO 理论上用 value network 做 credit assignment，但论文发现，在 reasoning-heavy 任务里，value network 往往不能可靠地区分中间状态好坏。

## 方法直觉

VinePPO 不训练 value network，而是在中间 step 之后继续采样多条 rollout，用未来成功率估计 state value：

```text
partial reasoning state s_t
  -> sample K continuations
  -> check final answer correctness
  -> estimate V(s_t) by success rate
  -> compute advantage for the step
  -> PPO-style update
```

它利用语言环境的特殊性：从任意 partial solution 继续生成是容易的，因此可以用额外 rollout 换更准确的 credit assignment。

## 和 PPO / GRPO 的区别

| 方法 | Credit assignment |
| --- | --- |
| PPO | learned value network 估 V(s) |
| GRPO / RLOO | response-level reward + group baseline |
| RestEM / rejection SFT | 正确整段都模仿，错误整段丢弃 |
| VinePPO | MC rollout 估计中间 state value |

VinePPO 的立场很微妙：它不接受 PPO 的 learned critic，但也不接受 GRPO 的“整段同权”。它说：critic 估不好，我们就用采样直接估。

## 流程图

```text
train trajectory:
  step_1 -> step_2 -> ... -> final answer

for selected intermediate state:
  state prefix
    -> K auxiliary continuations
    -> verifier scores final answers
    -> MC estimate of future success
    -> advantage for current step
```

## 关键实验

论文在 MATH 和 GSM8K 上比较 PPO、GRPO、RLOO、RestEM、DPO variants 等方法，报告 VinePPO 在测试准确率、KL tradeoff、generalization slope 上更好。虽然每轮可能因为 MC sampling 更慢，但它用更少训练步达到更高峰值，部分设置下 wall-clock 反而更省。

最有价值的观察是：value network 即使训练得看似不错，也可能在“比较两个候选中间步骤哪个更有前途”上接近随机。

## 和 PRM / PAV 的关系

VinePPO 和 PRM/PAV 都在解决 step-level credit assignment，但方式不同：

- PRM/PAV：训练一个 verifier / reward model，之后用它给 step 打分。
- VinePPO：不训练过程奖励模型，直接用 MC rollout 估计某个 state 的未来成功概率。

可以理解为：

```text
PRM/PAV = learned dense reward
VinePPO = sampled dense credit
```

两者未来可能结合：用 PRM/PAV 降低 VinePPO 的 MC 采样成本。

## 局限性

- MC rollout 成本高，长任务和大模型上会很贵。
- 需要能自动验证最终答案，开放任务不容易用。
- 中间步骤如何切分会影响估计质量。
- 如果 verifier 本身有漏洞，MC 估计只会更准确地优化这个漏洞。

## 对我理解这条路线的意义

VinePPO 是一个很重要的反身提醒：R1 后大家为了省显存去 critic，但不能把 credit assignment 也一起扔掉。长推理里，模型真正需要的是知道“哪一步让未来更可能成功”。

## 读这篇时抓住什么

抓住一句话：**value critic 不好，不等于 step credit 不重要。** VinePPO 的贡献就是把这两个问题拆开。

