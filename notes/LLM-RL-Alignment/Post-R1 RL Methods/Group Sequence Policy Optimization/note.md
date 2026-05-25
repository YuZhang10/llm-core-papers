---
title: "Group Sequence Policy Optimization"
paper_id: "arXiv:2507.18071"
authors: "Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, et al."
venue: "arXiv 2025"
date: "2026-05-21"
tags:
  - llm-rl
  - rlvr
  - gspo
  - grpo
  - sequence-level-rl
---

# Group Sequence Policy Optimization

## 本地文件

- PDF: [GSPO_arXiv_2507.18071.pdf](GSPO_arXiv_2507.18071.pdf)
- arXiv source: [GSPO_arXiv_2507.18071_source.tar.gz](GSPO_arXiv_2507.18071_source.tar.gz)
- arXiv source dir: [GSPO_arXiv_2507.18071_source](GSPO_arXiv_2507.18071_source)
- arXiv: https://arxiv.org/abs/2507.18071

## 一句话理解

GSPO 的核心质疑是：LLM reasoning 的 reward 通常是整段 response 的 outcome reward，那优化时为什么要用每个 token 的 importance ratio？它把 GRPO 的 token-level ratio 改成 sequence-level ratio，让 reward、clipping、optimization 的单位一致。

## 论文主线

GRPO / PPO-style LLM RL 常用 token-level ratio：

```text
w_t = pi_theta(y_t | x, y_<t) / pi_old(y_t | x, y_<t)
```

但 reasoning reward 往往长这样：

```text
response y  -> final answer correct? -> reward
```

也就是 reward 的单位是整段序列，而不是单个 token。GSPO 认为这个 mismatch 会让长序列训练产生高方差噪声，尤其在长 CoT 和 MoE 模型里更严重。

## 方法直觉

GSPO 直接定义 sequence-level importance ratio：

```text
s = pi_theta(y | x) / pi_old(y | x)
```

然后整段 response 一起做 clipping 和 optimization。直觉是：

- 如果整段回答相对 old policy 偏得太远，就整体 clip。
- 同一回答里的 tokens 共享 response-level advantage。
- 优化单位和 reward 单位一致，减少 token-level ratio 的随机波动。

## 为什么 token-level ratio 会出问题

长 CoT 里有成百上千个 token。某几个 token 的 log-prob 变化可能很大，但这并不一定对应整段回答质量变化。token-level clipping 会把局部概率噪声放大成训练噪声。

MoE 模型里还有 routing volatility：新旧 policy 计算 token likelihood 时可能激活不同 experts，导致 token-level ratio 更不稳定。GSPO 认为 sequence-level likelihood 对这类局部波动更鲁棒。

## 流程图

```text
prompt
  -> sample multiple responses from old policy
  -> verifier gives sequence rewards
  -> normalize rewards within group
  -> compute sequence likelihood ratio
  -> sequence-level clipping
  -> update policy
```

## 和 GRPO 的关键差异

| 维度 | GRPO | GSPO |
| --- | --- | --- |
| ratio 单位 | token | sequence |
| reward 单位 | sequence | sequence |
| clipping 单位 | token-wise | response-wise |
| 优势信号 | group-relative | group-relative |
| 主要风险 | token ratio 噪声、长序列不稳 | sequence ratio 估计与长度尺度敏感 |
| MoE 训练 | 可能需要 routing replay 等技巧 | 论文声称更自然稳定 |

## 关键实验与主张

论文声称 GSPO 相比 GRPO 有更好的训练效率、稳定性和 performance，尤其能稳定 MoE RL 训练，并对 Qwen3 最新模型的提升有贡献。

这里最重要的不是单个 benchmark 数字，而是工业信号：如果一个大模型家族把某个 RL update rule 放进训练 recipe，它通常说明这个设计解决了真实 scale 上的痛点。

## 和 DAPO / REINFORCE++ 的关系

- DAPO 是在 GRPO token-level 框架上加稳定训练技巧。
- GSPO 认为 token-level objective 本身就有问题，应该换成 sequence-level。
- REINFORCE++ 主要改 advantage normalization，从 local group 到 global batch。

三者可以看成分别动三个部位：

```text
DAPO: sampling / clip / length shaping
GSPO: importance ratio and clipping unit
REINFORCE++: advantage normalization estimator
```

## 局限性

- sequence-level ratio 对响应长度、概率乘积尺度很敏感，需要很小的 clipping range 和细致实现。
- 如果未来有可靠 token-level / step-level reward，纯 sequence-level update 可能不够细。
- 论文强依赖大规模训练经验，社区复现成本可能较高。
- 对非 MoE、小模型或短回答任务，收益未必同样明显。

## 对我理解这条路线的意义

GSPO 把一个很根本的问题摆出来：LLM RL 的 reward granularity 和 optimization granularity 必须匹配。R1 后大家都在谈 verifier 和 GRPO，但 GSPO 提醒我们，policy gradient 里的 importance ratio 选择本身就是大问题。

## 读这篇时抓住什么

抓住一句话：**reward 是整段级别，就不要让单个 token 的 ratio 主导整段更新。** 这句话是 GSPO 对 GRPO / DAPO 的核心批评。

