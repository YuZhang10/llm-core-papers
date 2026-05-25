---
title: "Constitutional AI: Harmlessness from AI Feedback"
short_title: "Constitutional AI / RLAIF"
paper_id: "arXiv:2212.08073"
authors: "Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al."
venue: "arXiv 2022"
date: "2026-05-21"
tags:
  - llm-rl
  - rlaif
  - constitutional-ai
  - scalable-oversight
---

# Constitutional AI: Harmlessness from AI Feedback

## 本地文件

- PDF: [Constitutional_AI_arXiv_2212.08073.pdf](Constitutional_AI_arXiv_2212.08073.pdf)
- arXiv source: [Constitutional_AI_arXiv_2212.08073_source.tar.gz](Constitutional_AI_arXiv_2212.08073_source.tar.gz)
- arXiv source dir: [Constitutional_AI_arXiv_2212.08073_source](Constitutional_AI_arXiv_2212.08073_source)
- arXiv: https://arxiv.org/abs/2212.08073
- citation snapshot: Emergent Mind / Semantic Scholar surface 1,202（2026-05-21 检索）

## 一句话理解

Constitutional AI 的核心是把“人类逐条告诉模型哪个回答更好”改成“人类写一组原则，让 AI 按原则自我批评、自我修改、再用 AI feedback 做 RL”，从而大幅减少 harmlessness 对齐所需的人类标签。

## 论文主线

人工偏好标注贵，而且越到安全和伦理边界越难规模化。作者提出的替代方案是：人类主要提供一份 constitution，也就是原则列表。模型用这些原则监督自己。

流程分两段：

```text
SL phase:
  harmful prompt -> initial response
  -> model critiques response using constitution
  -> model revises response
  -> finetune on revised responses

RL phase:
  sample response pairs from SL-CAI model
  -> AI judges which response better follows constitution
  -> train preference model on AI preferences
  -> RL optimize with that preference model
```

这就是 RLAIF: Reinforcement Learning from AI Feedback。

## 方法直觉

### Constitution 把监督从样本层提升到原则层

在人类 RLHF 中，人类要对大量 response pair 做选择。在 CAI 中，人类先写原则，例如避免伤害、尊重自主、不要提供违法操作细节。之后每个样本怎么改、哪条回答更好，主要由 AI 根据原则来判断。

### Self-critique 先修行为分布

SL phase 很关键。模型先生成不安全回答，再按 constitution 批评自己并修订。这相当于自动构造一批“从坏回答到较好回答”的训练数据，让模型先学到一种非回避但更安全的回答风格。

### RLAIF 再做偏好强化

RL phase 不是直接拿 constitution 当 reward function，而是用 AI preference 数据训练 PM，再用 RL 优化。它和 HH-RLHF 的结构相似，只是 preference label 的来源从人类变成 AI。

## 关键实验与现象

- CAI 能在减少人类 harmlessness 标签的情况下，提高模型的 harmlessness。
- 作者强调“harmless but non-evasive”：目标不是一律拒答，而是在危险请求上说明边界、给出安全替代。
- Chain-of-thought style critique 可以帮助模型生成更透明的判断过程，但这也带来“模型是否真的按原则判断”的问题。

## 和相关工作的关系

- 相对 HH-RLHF：它主要替换反馈来源，降低人类标注成本。
- 相对 DPO：DPO 改的是优化目标，CAI 改的是偏好数据生产方式。两者可以组合。
- 相对 DeepSeekMath / GRPO：CAI 的 reward 仍是偏好代理，DeepSeekMath 更多依靠可验证答案。

## 局限性

- Constitution 本身来自人类选择，仍然带有价值判断和覆盖不足的问题。
- AI feedback 可能继承基础模型的偏见、误判和盲点。
- 当模型能力低于任务难度时，让 AI 监督 AI 可能只是放大错误。
- RLAIF 减少了人工标签，不等于消除了人类监督需求；原则设计、红队评测和上线监控仍然关键。

## 对我理解这条路线的意义

这篇把“可扩展监督”具体落到了 LLM alignment 里：人不再只做数据标注员，而是写规则、设计评测、审计系统。后来的 AI feedback、self-rewarding、LLM-as-judge、自动红队，都可以看作对这个想法的不同展开。

## 读这篇时抓住什么

重点不是 constitution 里的每条原则，而是监督接口的变化：从 human labels 到 human-written principles，再到 AI-generated critiques/preferences。这个变化同时带来效率和风险：规模上去了，反馈源的可靠性问题也被放大了。

