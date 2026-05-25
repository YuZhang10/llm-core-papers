---
title: "Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning"
paper_id: "arXiv:2410.08146"
authors: "Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, Aviral Kumar"
venue: "arXiv 2024"
date: "2026-05-21"
tags:
  - llm-rl
  - process-reward
  - process-advantage-verifier
  - pav
  - credit-assignment
---

# Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning

## 本地文件

- PDF: [Rewarding_Progress_arXiv_2410.08146.pdf](Rewarding_Progress_arXiv_2410.08146.pdf)
- arXiv source: [Rewarding_Progress_arXiv_2410.08146_source.tar.gz](Rewarding_Progress_arXiv_2410.08146_source.tar.gz)
- arXiv source dir: [Rewarding_Progress_arXiv_2410.08146_source](Rewarding_Progress_arXiv_2410.08146_source)
- arXiv: https://arxiv.org/abs/2410.08146

## 一句话理解

这篇把 PRM 的问题问得更精确：过程奖励不应该只是判断某一步“看起来对不对”，而应该奖励这一步带来的 progress，也就是让未来成功概率增加了多少。作者把这种模型叫 Process Advantage Verifier (PAV)。

## 论文主线

传统 PRM 常被理解为 step correctness classifier：

```text
step is correct / incorrect / neutral
```

但这篇认为，对 search 或 RL 来说，更重要的是：

```text
after this step, is the policy more likely to eventually solve the problem?
```

也就是 process reward 应该像 RL advantage：

```text
progress = success likelihood after step - success likelihood before step
```

## 方法直觉

一个步骤可能数学上没错，但没有带来进展，例如只是重述题目。反过来，一个步骤可能不完整，但把状态带到更容易继续求解的位置。

PAV 的目标不是给步骤打“道德分”或“格式分”，而是估计它是否让未来更可能成功。

## Prover policy 的角色

论文强调 progress 应该在一个 prover policy 下测量，而且这个 prover 可以不同于 base policy。

直觉是：

- 如果用 base policy 自己测，它本来不会解，很多有用步骤也可能看不出来。
- 如果用过强 prover，它从任何状态都能解，很多步骤的差异又被抹平。
- 好的 prover 需要和 base policy 有互补性，能区分 base policy 生成步骤的好坏。

这是这篇相对普通 PRM 论文最有意思的地方：process reward 不只是“训练一个更强 verifier”，还要问 verifier 的 reference policy 是谁。

## 流程图

```text
base policy generates partial reasoning state
        |
        v
candidate step
        |
        v
prover estimates success before / after step
        |
        v
process advantage = progress
        |
        v
PAV learns to predict this progress
        |
        v
use PAV for test-time search or online RL dense reward
```

## 关键结果

论文报告 PAV 相比 ORM：

- test-time search 更准确，且更 compute-efficient。
- online RL 中作为 dense reward 可以显著提升 sample efficiency。
- 过程奖励如果设计成 absolute Q-value，可能保留“看起来有希望但当前 step 没进展”的状态；advantage/progress 更适合做探索信号。

这些结果把 PRM 从“每步正确性评估”推进到“每步进展评估”。

## 和 Let's Verify Step by Step 的关系

| 论文 | 过程奖励关注点 |
| --- | --- |
| Let's Verify Step by Step | 人类标注每一步是否正确 |
| Rewarding Progress | 自动估计每一步是否提升未来成功概率 |

两者不是冲突关系。PRM800K 提供了过程监督基础设施；PAV 则重新定义“好的过程奖励应该是什么”。

## 和 VinePPO 的关系

VinePPO 直接用 MC rollout 估中间 state value；PAV 训练一个 verifier 去预测 progress。两者可以看成一动一静：

```text
VinePPO: online sampling estimates value
PAV: learned model predicts progress
```

如果 PAV 足够准，它可以减少 VinePPO 式 MC 采样成本。

## 局限性

- 如何自动选择好的 prover policy 仍是开放问题。
- PAV 训练误差会直接影响 search / RL。
- 数学 reasoning 上可行，不代表开放 agent 轨迹也容易定义 progress。
- 如果 progress verifier 被 policy exploit，仍会出现 reward hacking。

## 对我理解这条路线的意义

这篇把“过程奖励”从静态评语变成动态差分：不是问这一步好不好看，而是问这一步有没有让未来更容易成功。这对长推理和 agent RL 都非常关键。

## 读这篇时抓住什么

抓住一句话：**process reward 应该奖励状态改善，而不只是奖励步骤正确。** 这是 PAV 和传统 PRM 的分水岭。

