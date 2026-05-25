---
title: "Training language models to follow instructions with human feedback"
short_title: "InstructGPT / RLHF"
paper_id: "arXiv:2203.02155"
authors: "Long Ouyang, Jeff Wu, Xu Jiang, et al."
venue: "NeurIPS 2022"
date: "2026-05-21"
tags:
  - llm-rl
  - rlhf
  - instructgpt
  - preference-alignment
---

# Training language models to follow instructions with human feedback

## 本地文件

- PDF: [InstructGPT_RLHF_arXiv_2203.02155.pdf](InstructGPT_RLHF_arXiv_2203.02155.pdf)
- arXiv source: [InstructGPT_RLHF_arXiv_2203.02155_source.tar.gz](InstructGPT_RLHF_arXiv_2203.02155_source.tar.gz)
- arXiv source dir: [InstructGPT_RLHF_arXiv_2203.02155_source](InstructGPT_RLHF_arXiv_2203.02155_source)
- arXiv: https://arxiv.org/abs/2203.02155
- citation snapshot: Emergent Mind / Semantic Scholar surface 9,967（2026-05-21 检索）

## 一句话理解

这篇就是现代 LLM RLHF 的基准模板：先用人工示范做 SFT，再用人工偏好训练 reward model，最后用 PPO 在 KL 约束下优化模型，让 GPT-3 从“会续写互联网文本”变成“更愿意按用户意图回答”的 InstructGPT。

## 论文主线

作者要解决的问题很直接：模型变大并不会自动更听话。GPT-3 会续写、会补全，但用户真正想要的是有帮助、真实、低毒性的回答。预训练目标是 next-token prediction，它和“按人类意图做事”之间有明显错位。

论文给出的路线是三段式：

```text
labeler-written/API prompts
        -> human demonstrations
        -> supervised fine-tuning (SFT)
        -> model samples
        -> human rankings
        -> reward model (RM)
        -> PPO with KL penalty
        -> InstructGPT
```

SFT 负责把模型拉到 instruction-following 的基本分布；reward model 把人类偏好压成可优化的标量；PPO 负责在不偏离参考模型太远的前提下提高 reward。

## 方法直觉

### SFT 不是终点，而是起跑线

只做 SFT 可以让模型学会“像标注者那样回答”，但它没有直接优化“两个答案哪个更好”。如果标注数据有限，SFT 很容易学到表层风格。RLHF 的关键是让模型见到多个候选回答之间的偏好差异。

### Reward model 是人类偏好的代理

训练数据不是单个分数，而是同一 prompt 下多个回答的排序。reward model 学的是“chosen 比 rejected 更好”的概率。它把复杂的主观判断压缩成一个 reward，让 PPO 可以用强化学习方式继续推模型。

### PPO 的 KL 惩罚是保险丝

如果只最大化 reward model，模型可能 exploit reward model，生成高分但古怪的回答。论文用 KL penalty 把 policy 限制在 SFT/reference model 附近，相当于说：可以朝人类偏好移动，但不要完全离开语言模型原本可靠的区域。

## 关键实验

- 人类评估里，1.3B InstructGPT 的输出可以被偏好于 175B GPT-3，说明“对齐目标”在用户体验上比参数规模更直接。
- InstructGPT 在 TruthfulQA、toxicity 等维度上改善，同时对传统 NLP benchmark 的退化较小。
- 论文把数据来源、标注流程、reward model、PPO 训练合成一个可复用 pipeline，这也是它引用量极高的原因。

## 和后续工作的关系

- HH-RLHF 沿着这条路线把 assistant 的 helpfulness / harmlessness 做得更系统。
- Constitutional AI 把“人工偏好”替换为“原则 + AI feedback”。
- DPO 重新解释这条 KL-regularized RLHF 目标，绕开显式 reward model 和 PPO。
- GRPO / DeepSeekMath 保留 online rollout 和 KL/reference 思想，但把奖励换成可验证 reasoning reward，并去掉 value critic。

## 局限性

- 人类偏好昂贵，且偏好标准会随标注者、任务、文化而变化。
- Reward model 是代理目标，过度优化会导致 reward hacking 或长度偏置。
- PPO 工程复杂，需要 policy、reference、reward model，有时还需要 value model，训练稳定性敏感。
- 论文的 prompt 分布来自 API 和标注者构造，真实用户分布变化时仍可能失效。

## 读这篇时抓住什么

这篇不要只记“用了 PPO”。真正要记的是：LLM 后训练第一次被组织成了可操作的偏好学习系统。SFT、RM、PPO、KL penalty 这四块后来不断被替换，但问题设置基本没变：如何把模糊的人类偏好变成稳定、可扩展、不会把模型训坏的优化信号。

