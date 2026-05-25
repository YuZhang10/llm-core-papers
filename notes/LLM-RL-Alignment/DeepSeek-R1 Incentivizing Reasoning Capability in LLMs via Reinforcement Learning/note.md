---
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
short_title: "DeepSeek-R1"
paper_id: "arXiv:2501.12948"
authors: "DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, et al."
venue: "arXiv 2025"
date: "2026-05-21"
tags:
  - llm-rl
  - reasoning-rl
  - rlvr
  - grpo
  - deepseek
---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## 本地文件

- PDF: [DeepSeek_R1_arXiv_2501.12948.pdf](DeepSeek_R1_arXiv_2501.12948.pdf)
- arXiv source: [DeepSeek_R1_arXiv_2501.12948_source.tar.gz](DeepSeek_R1_arXiv_2501.12948_source.tar.gz)
- arXiv source dir: [DeepSeek_R1_arXiv_2501.12948_source](DeepSeek_R1_arXiv_2501.12948_source)
- arXiv: https://arxiv.org/abs/2501.12948
- Hugging Face paper page: https://huggingface.co/papers/2501.12948
- citation note: 这篇影响极大，但本轮没有找到稳定的公开破千引用快照；作为 GRPO / RLVR 延伸读物放入专题，不计入 README 的严格主清单。

## 一句话理解

DeepSeek-R1 把 DeepSeekMath/GRPO 这条线推到大众视野：先展示 R1-Zero 可以直接用大规模 RL 从 base model 中激发长思考、自检、反思等 reasoning 行为，再用 cold-start data、多阶段 RL、rejection sampling 和 distillation 把这种能力整理成可用模型。

## 论文主线

R1 论文有两个层次：

1. **R1-Zero**：不用 supervised fine-tuning 作为起点，直接对 base model 做 reasoning-oriented RL。
2. **R1**：发现 pure RL 虽然能涌现推理，但可读性、语言一致性和通用可用性不足，于是加入 cold-start SFT 和多阶段训练。

大致流程：

```text
R1-Zero:
  base model -> RL with verifiable / rule rewards -> emergent long reasoning

R1:
  cold-start reasoning data
  -> reasoning RL
  -> rejection sampling + SFT data
  -> second RL stage for helpfulness / harmlessness / reasoning
  -> distilled smaller models
```

## 方法直觉

### Pure RL 的意义

R1-Zero 最吸引人的地方不是某个 benchmark 数字，而是叙事上的反转：复杂 reasoning 行为不一定只能从人工 CoT 数据中模仿出来，也可以通过可验证 reward 和大规模 rollout 被强化出来。

论文报告了训练中出现的自我反思、重新检查、延长思考等行为。这些行为看起来像“模型学会了思考更久”，但更谨慎地说，是 policy 学会了在 reward 有利的情况下生成更长、更结构化、更容易到达正确答案的推理轨迹。

### Cold-start 的意义

Pure RL 会带来可读性差、语言混杂、格式不稳定等问题。R1 加 cold-start data，是为了先给模型一个更可读的 reasoning 分布，再让 RL 放大正确推理。这个设计说明：RL 能激发能力，但 SFT 仍然是控制输出形态的强工具。

### Distillation 的意义

R1 的另一个影响点是把大模型 reasoning 轨迹蒸馏到较小 dense model。对开源社区来说，直接复现大规模 RL 成本很高，但用高质量 reasoning data 蒸馏更可操作。

## 和 DeepSeekMath / GRPO 的关系

DeepSeekMath 提出 GRPO 并展示数学 reasoning 提升；R1 把这个思路扩大成 reasoning model 训练范式。可以把二者关系理解为：

```text
DeepSeekMath: data + GRPO improves math reasoning in a 7B model
DeepSeek-R1: large-scale RLVR + staged training builds general reasoning models
```

R1 之后，社区里的 DAPO、GSPO、Reinforce++、VinePPO、process reward model、agent RL 等工作，很多都在回答同一个问题：如何让 RL 在长链 reasoning 中更稳定、更高效、更不依赖黑盒 recipe。

## 关键观察

- RL 可以显著提高 reasoning benchmark 表现，并增加 response length / test-time compute。
- Pure RL 会自然产生某些“反思式”行为，但不一定满足产品可读性和多语言稳定性。
- 多阶段训练不是装饰，而是把“能力激发”和“行为整理”分开处理。
- 小模型蒸馏说明 reasoning capability 的一部分可以通过数据传递，而不一定每个模型都要从头跑昂贵 RL。

## 局限性

- 技术报告没有完全公开所有训练细节，严格复现仍然困难。
- 可验证 reward 主要适用于数学、代码等有明确答案的领域，开放式任务更难。
- 长推理不等于真实可靠推理，可能只是 reward 下的高分文本模式。
- Safety / harmlessness 阶段和 reasoning RL 之间可能有张力，尤其在模型更会规划之后。

## 读这篇时抓住什么

不要只把 R1 读成“GRPO 很强”。更准确的理解是：R1 把 LLM 后训练从偏好对齐扩展到能力激发，尤其是可验证任务上的 reasoning policy optimization。它把 2022 的 RLHF 问题改写成 2025 的新问题：当 reward 可验证、rollout 足够多、模型已有潜在解法空间时，RL 到底能把哪些能力从模型里逼出来？

