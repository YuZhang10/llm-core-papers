---
title: "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"
short_title: "HH-RLHF"
paper_id: "arXiv:2204.05862"
authors: "Yuntao Bai, Andy Jones, Kamal Ndousse, et al."
venue: "arXiv 2022"
date: "2026-05-21"
tags:
  - llm-rl
  - rlhf
  - helpful-harmless
  - anthropic
---

# Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback

## 本地文件

- PDF: [HH_RLHF_arXiv_2204.05862.pdf](HH_RLHF_arXiv_2204.05862.pdf)
- arXiv source: [HH_RLHF_arXiv_2204.05862_source.tar.gz](HH_RLHF_arXiv_2204.05862_source.tar.gz)
- arXiv source dir: [HH_RLHF_arXiv_2204.05862_source](HH_RLHF_arXiv_2204.05862_source)
- arXiv: https://arxiv.org/abs/2204.05862
- citation snapshot: Emergent Mind / Semantic Scholar surface 1,816；搜索面另见 3,729（2026-05-21 检索）

## 一句话理解

这篇把 RLHF 从“让模型更听指令”推进到“训练一个 helpful 且 harmless 的通用助手”：用人类比较数据训练 preference model，再用 RL 优化 assistant，并研究在线迭代、偏好模型规模、KL 偏移和安全目标之间的关系。

## 论文主线

InstructGPT 更关注 instruction-following，这篇更关注 assistant alignment。作者把目标拆成两个看似可能冲突的方向：

- helpful：回答要有用、相关、能解决用户问题。
- harmless：面对危险、违法、恶意、自伤等请求时不要直接配合。

论文的主线是：这两个目标不是只能二选一。随着模型和 preference model 变强，helpfulness 与 harmlessness 可以在同一 RLHF 框架下共同改进。

## 方法直觉

### 偏好数据来自对话，而不是静态分类

标注者和模型进行多轮对话，然后在同一上下文下比较两个候选回复。这个设置比单轮 prompt 更接近真实 assistant。数据分为 helpfulness 和 harmlessness 两类，分别训练或混合训练 preference model。

### Preference model 是可扩展监督的核心

作者系统研究了 preference model 的规模：PM 越强，越能区分微妙的好坏回答。RL 阶段不是直接听人类，而是听 PM，因此 PM 质量决定了后续 RL 的上限和偏差。

### 在线迭代很重要

如果 policy 变了，旧偏好数据就会越来越 off-policy。作者探索每周更新数据、更新 PM、更新 RL policy 的在线迭代模式。这一点后来在 online RLHF、iterative DPO、self-training 中反复出现。

## RL 视角

这篇对 RLHF 的一个重要观察是：reward 提升和 policy 偏离初始化模型之间存在近似关系。换句话说，RL 不是免费午餐。你越用 reward model 推模型，模型离原来的语言分布越远，越需要 KL / reference 约束来防止怪异行为。

可以这样理解：

```text
pretrained / context-distilled assistant
        -> collect pairwise preferences
        -> train helpful PM and harmless PM
        -> PPO optimize PM reward with KL control
        -> collect fresher comparisons
        -> iterate
```

## 关键实验

- RLHF 在许多 NLP 评测上没有明显 performance tax，有时反而有提升。
- helpful-only 和 harmless-only 的目标会有张力，但混合训练和更强模型能缓解冲突。
- 在线迭代偏好数据可以让模型持续改善，而不是只在固定离线数据上优化。
- PM calibration、OOD detection、competing objectives 等分析让这篇更像 RLHF 系统论文，而不是单一算法论文。

## 和相关工作的关系

- 它继承 InstructGPT 的 RM + PPO 管线，但更强调通用助手和安全目标。
- Constitutional AI 是对这篇人工反馈成本的直接回应：能不能少用人工 harmlessness 标签？
- DPO 类方法会保留 pairwise preference 数据，但去掉显式 PM + PPO。
- DeepSeekMath / R1 的 reasoning RL 则把 reward 从“人类偏好代理”换成“答案可验证信号”。

## 局限性

- Harmlessness 的定义依赖标注规范，跨文化、跨应用很难统一。
- PM 不是人类本身，可能学到长度、礼貌、拒答风格等捷径。
- 在线迭代虽好，但真实线上人类反馈成本高、延迟大，开源社区很难复现。
- RLHF 仍可能在边界场景上产生 reward hacking 或过度拒答。

## 读这篇时抓住什么

这篇的价值在于把 RLHF 当成一个长期运行的对齐系统来看，而不是一次性 fine-tuning trick。读的时候重点看三件事：偏好数据怎么随 policy 演化，PM 规模如何决定监督质量，以及 KL 控制如何在 reward 最大化和保持语言能力之间做平衡。
