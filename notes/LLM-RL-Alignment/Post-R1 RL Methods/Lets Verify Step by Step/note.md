---
title: "Let's Verify Step by Step"
paper_id: "arXiv:2305.20050"
authors: "Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe"
venue: "arXiv 2023"
date: "2026-05-21"
tags:
  - llm-rl
  - process-supervision
  - prm
  - prm800k
  - mathematical-reasoning
---

# Let's Verify Step by Step

## 本地文件

- PDF: [Lets_Verify_Step_by_Step_arXiv_2305.20050.pdf](Lets_Verify_Step_by_Step_arXiv_2305.20050.pdf)
- arXiv source: [Lets_Verify_Step_by_Step_arXiv_2305.20050_source.tar.gz](Lets_Verify_Step_by_Step_arXiv_2305.20050_source.tar.gz)
- arXiv source dir: [Lets_Verify_Step_by_Step_arXiv_2305.20050_source](Lets_Verify_Step_by_Step_arXiv_2305.20050_source)
- arXiv: https://arxiv.org/abs/2305.20050
- Dataset: https://github.com/openai/prm800k

## 一句话理解

这篇是 PRM 路线的基础论文：与其只监督最终答案对不对，不如监督每一步推理是否正确。它发布了 PRM800K，并显示 process supervision 在 MATH 上比 outcome supervision 更有效。

## 论文主线

数学推理常见问题是：最终答案可能对，但推理过程错；最终答案错，也不知道哪一步错。

论文比较两种 reward model 训练方式：

```text
Outcome Reward Model (ORM):
  full solution -> final answer correct? -> label

Process Reward Model (PRM):
  step-by-step solution -> each step correct / incorrect / neutral -> labels
```

作者强调：这篇主要训练 reward model，不是直接训练 generator 做 RL。generator 固定，PRM/ORM 用来给生成解排序。

## 数据：PRM800K

PRM800K 包含约 800K 个 step-level human feedback labels，来自约 75K 个 solution samples 和 12K 个 problems。标注者对每一步给出：

- positive：正确且合理。
- negative：错误或不合理。
- neutral：有歧义或难判断。

标注策略里有一个重要设计：对错误解通常只标到第一个错误步骤。这让 process supervision 和 outcome supervision 的成本更可比，因为人类只需要定位第一个错点。

## 方法直觉

Outcome supervision 的信息很粗：

```text
这整段解法错了。
```

Process supervision 的信息更细：

```text
前 3 步还可以，第 4 步开始错。
```

这对 reward model 来说是巨大差异。特别是在 hard MATH problems 里，大部分候选解都会错，单纯负标签提供的信息很少；而第一个错点可以直接告诉模型哪里失去可靠性。

## 流程图

```text
model generates step-by-step solutions
        |
        v
human labels each step
        |
        v
train PRM to score prefixes / steps
        |
        v
best-of-N search selects solution using PRM
```

## 关键实验

论文报告 large-scale PRM 在代表性 MATH test subset 上 best-of-N 可以解出 78.2% 的问题，高于 ORM 和 majority voting。它还显示 active learning 让 process supervision 的数据效率提升约 2.6x。

这里的核心现象是：PRM 不只是更安全、更可解释，它在最终 outcome 上也更强。这就是论文里所谓 negative alignment tax 的味道：更对齐的监督方式反而提升能力。

## 和后续工作的关系

- Rewarding Progress / PAV 继续追问：process reward 应该奖励“步骤正确”，还是奖励“让未来更可能成功”？
- VinePPO 用 MC rollout 估 step value，不依赖人工 PRM 标注。
- Agent RL 里的 long-horizon credit assignment，很可能也需要 PRM 类 dense feedback。

## 局限性

- Step-level human labels 很贵，规模化困难。
- 数学步骤相对可审查，开放任务或 agent 轨迹更难定义“正确步骤”。
- PRM 用于 search 很自然，用于 online RL 时还要处理 reward hacking 和分布偏移。
- PRM 可能偏好看起来规范的推理，而不一定真正证明 reasoning faithful。

## 对我理解这条路线的意义

这篇把 RLVR 从“最终答案 verifier”推进到“过程 verifier”。如果 outcome reward 是最终考试分数，PRM 就像老师在草稿纸上批改每一步。它直接打开了 process reward、test-time search、step-level RL 的路线。

## 读这篇时抓住什么

抓住这组对比：**ORM 解决结果判断，PRM 解决错误定位。** 长推理想继续提升，错误定位往往比最终判分更关键。

