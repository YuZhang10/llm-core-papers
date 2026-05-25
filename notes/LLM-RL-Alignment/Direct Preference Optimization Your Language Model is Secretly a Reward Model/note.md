---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
short_title: "DPO"
paper_id: "arXiv:2305.18290"
authors: "Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn"
venue: "NeurIPS 2023"
date: "2026-05-21"
tags:
  - llm-rl
  - dpo
  - preference-optimization
  - direct-alignment
---

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

## 本地文件

- PDF: [DPO_arXiv_2305.18290.pdf](DPO_arXiv_2305.18290.pdf)
- arXiv source: [DPO_arXiv_2305.18290_source.tar.gz](DPO_arXiv_2305.18290_source.tar.gz)
- arXiv source dir: [DPO_arXiv_2305.18290_source](DPO_arXiv_2305.18290_source)
- arXiv: https://arxiv.org/abs/2305.18290
- NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
- citation snapshot: Semantic Scholar surface 7,658（2026-05-21 检索）

## 一句话理解

DPO 发现 KL-regularized RLHF 里的 reward 和最优 policy 存在闭式关系，于是可以不显式训练 reward model、不跑 PPO，直接用 chosen / rejected 的 log probability ratio 做一个 preference classification loss。

## 论文主线

标准 RLHF 通常是：

```text
pretrained model -> SFT policy
preference data -> reward model
policy + reward model -> PPO with KL to reference
```

DPO 问的是：既然最终要优化的是 policy，能不能跳过 reward model 和 PPO？

关键推导来自 KL-regularized reward maximization。对一个 prompt $x$，最优策略大致满足：

```text
pi*(y | x) proportional to pi_ref(y | x) * exp(r(x, y) / beta)
```

反过来，reward 可以由 policy 和 reference policy 的 log-ratio 表示：

```text
r(x, y) = beta * (log pi(y | x) - log pi_ref(y | x)) + constant
```

把这个隐式 reward 放进 Bradley-Terry preference model，就得到 DPO loss。

## 方法直觉

### Policy 自己就是 reward model

如果某个回答在当前 policy 下比 reference policy 更容易生成，DPO 会把它解释成“当前 policy 认为这个回答 reward 更高”。chosen / rejected 的差异就变成两个 log-ratio 的差异。

### DPO 把 RL 问题变成分类问题

训练样本是 `(prompt, chosen, rejected)`。目标是让 chosen 的隐式 reward 高于 rejected。工程上就是一次普通的 supervised fine-tuning 风格训练，不需要 rollout、不需要在线采样、不需要 value critic。

### beta 控制对齐强度

beta 越小，模型越激进地偏离 reference；beta 越大，越保守。它对应 RLHF 里的 KL regularization 强度，是 DPO 里最关键的旋钮之一。

## 关键贡献

- 给出了从 KL-regularized RLHF 到 direct preference loss 的清晰推导。
- 把原本复杂的 RM + PPO pipeline 简化成单阶段离线训练。
- 在摘要、对话、情感控制等任务上展示了和 RLHF 相当或更好的效果。
- 直接引爆了后续直接偏好优化家族：IPO、KTO、ORPO、CPO、SimPO、NCA/DNO 等。

## 和 RLHF 的关系

DPO 不是“完全没有 RL 思想”。它是把 RLHF 的最优解结构代入 preference likelihood 后得到的离线目标。它保留了 reference policy、KL 控制、pairwise preference 这些核心概念，只是把优化器从 PPO 换成了 supervised classification。

可以这样对照：

| 维度 | PPO-RLHF | DPO |
| --- | --- | --- |
| Reward model | 显式训练 | 隐式由 policy/reference log-ratio 表示 |
| Sampling | 通常 on-policy rollout | 离线 preference pairs |
| Value critic | 常需要 | 不需要 |
| 工程稳定性 | 超参敏感 | 更接近 SFT |
| 风险 | RM hacking、PPO 不稳定 | 数据偏差、beta 敏感、长度偏置 |

## 局限性

- DPO 依赖离线 preference 数据，数据覆盖不到的区域很难通过探索发现。
- Bradley-Terry 偏好假设和 KL-regularized optimal policy 假设未必总成立。
- 如果 chosen 比 rejected 长很多，log probability 累积方式可能引入长度偏置。
- DPO 没有显式 reward model，不代表没有 reward hacking，只是 reward 被隐式塞进了 policy ratio。

## 读这篇时抓住什么

DPO 的关键不是“分类 loss 很简单”，而是它给了一个统一视角：很多 LLM alignment 目标都可以看成在 reference model 附近重塑概率分布。读懂这篇之后，再看 IPO / KTO / SimPO / GRPO，会更容易分清它们到底是在改 reward 假设、偏好模型、reference 项，还是采样方式。

