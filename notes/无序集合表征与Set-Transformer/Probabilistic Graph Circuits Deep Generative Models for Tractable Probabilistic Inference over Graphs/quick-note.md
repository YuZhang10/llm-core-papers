## Probabilistic Graph Circuits: Deep Generative Models for Tractable Probabilistic Inference over Graphs

### 决策卡片
- 年份：2025
- 引用数：1（Semantic Scholar）
- 与搜索意图相关性：中 —— 论文明确讨论图上的 permutation invariance，并涉及深度生成模型，但核心是 probabilistic graph circuits 与可精确推断，不是 Deep Sets 或 Transformer 架构本身。
- 是否值得进入精读候选：可作为次级候选；如果你关注“置换不变性如何影响图生成/概率推断”值得看，若目标是 Deep Sets/Transformer 的通用 permutation-invariant 表示学习，则优先级不高。

### 摘要原文

Deep generative models (DGMs) have recently demonstrated remarkable success in capturing complex probability distributions over graphs. Although their excellent performance is attributed to powerful and scalable deep neural networks, it is, at the same time, exactly the presence of these highly non-linear transformations that makes DGMs intractable. Indeed, despite representing probability distributions, intractable DGMs deny probabilistic foundations by their inability to answer even the most basic inference queries without approximations or design choices specific to a very narrow range of queries. To address this limitation, we propose probabilistic graph circuits (PGCs), a framework of tractable DGMs that provide exact and efficient probabilistic inference over (arbitrary parts of) graphs. Nonetheless, achieving both exactness and efficiency is challenging in the permutation-invariant setting of graphs. We design PGCs that are inherently invariant and satisfy these two requirements, yet at the cost of low expressive power. Therefore, we investigate two alternative strategies to achieve the invariance: the first sacrifices the efficiency, and the second sacrifices the exactness. We demonstrate that ignoring the permutation invariance can have severe consequences in anomaly detection, and that the latter approach is competitive with, and sometimes better than, existing intractable DGMs in the context of molecular graph generation.

### 摘要中文翻译

深度生成模型近年来在刻画图上的复杂概率分布方面取得了显著成功。虽然这些优异表现归功于强大且可扩展的深度神经网络，但与此同时，正是这些高度非线性变换使得深度生成模型难以进行可处理的推断。事实上，尽管它们表示的是概率分布，不可处理的深度生成模型却无法在不依赖近似方法或针对极窄查询范围的特定设计选择的情况下回答最基本的推断问题，从而削弱了其概率基础。

为了解决这一限制，作者提出了 probabilistic graph circuits，简称 PGCs。这是一个可处理的深度生成模型框架，能够对图的任意部分进行精确且高效的概率推断。然而，在图的置换不变设置下，同时实现精确性和效率具有挑战。作者设计了天然满足不变性的 PGC，并满足精确和高效这两个要求，但代价是表达能力较低。因此，作者进一步研究了两种实现不变性的替代策略：第一种牺牲效率，第二种牺牲精确性。作者展示了忽略置换不变性在异常检测中可能带来严重后果，并表明后一种方法在分子图生成任务中可以与现有不可处理的深度生成模型竞争，有时甚至表现更好。

### 这篇论文大概在解决什么

这篇论文关注的是：如何在图数据的深度生成建模中，同时考虑概率推断的可处理性和图结构的置换不变性。

它不是主要提出 Deep Sets 或 Transformer-style permutation invariant network，而是从概率图生成模型/概率电路角度切入，试图让模型既能表示图分布，又能对图的部分结构做精确、高效推断。摘要中特别强调，图的 permutation invariance 会和“精确推断”“高效推断”“表达能力”之间产生权衡。

### 可能需要精读时重点看什么

- PGCs 如何定义图上的概率分布，以及为什么能实现 tractable inference。
- 作者所说的“inherently invariant” PGC 是如何保证 permutation invariance 的。
- 精确性、效率、表达能力三者之间的 trade-off 具体是什么。
- 两种替代置换不变策略分别牺牲了什么：一种牺牲效率，一种牺牲精确性。
- 与 Deep Sets/Transformer 相关性有限，但可重点关注它对 permutation invariance 的形式化处理，尤其是图生成和异常检测场景中的影响。
