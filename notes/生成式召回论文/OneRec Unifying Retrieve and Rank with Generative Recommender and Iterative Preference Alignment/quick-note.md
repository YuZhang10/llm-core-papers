## OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment

### 决策卡片
- 年份：2025
- 引用数：193
- 与搜索意图相关性：高 —— 论文明确以 generative retrieval-based recommendation 为背景，目标是用统一生成式模型替代传统 retrieve-and-rank 级联链路，属于工业级生成式召回/推荐的核心候选。
- 是否值得进入精读候选：非常值得。适合重点看“召回排序一体化”“session-wise generation”“生成式推荐线上部署”和“偏好对齐如何进入推荐召回”。

### 摘要原文

Recently, generative retrieval-based recommendation systems have emerged as a promising paradigm. However, most modern recommender systems adopt a retrieve-and-rank strategy, where the generative model functions only as a selector during the retrieval stage. In this paper, we propose OneRec, which replaces the cascaded learning framework with a unified generative model. To the best of our knowledge, this is the first end-to-end generative model that significantly surpasses current complex and well-designed recommender systems in real-world scenarios. Specifically, OneRec includes: 1) an encoder-decoder structure, which encodes the user's historical behavior sequences and gradually decodes the videos that the user may be interested in. We adopt sparse Mixture-of-Experts (MoE) to scale model capacity without proportionally increasing computational FLOPs. 2) a session-wise generation approach. In contrast to traditional next-item prediction, we propose a session-wise generation, which is more elegant and contextually coherent than point-by-point generation that relies on hand-crafted rules to properly combine the generated results. 3) an Iterative Preference Alignment module combined with Direct Preference Optimization (DPO) to enhance the quality of the generated results. Unlike DPO in NLP, a recommendation system typically has only one opportunity to display results for each user's browsing request, making it impossible to obtain positive and negative samples simultaneously. To address this limitation, We design a reward model to simulate user generation and customize the sampling strategy. Extensive experiments have demonstrated that a limited number of DPO samples can align user interest preferences and significantly improve the quality of generated results. We deployed OneRec in the main scene of Kuaishou, achieving a 1.6\% increase in watch-time, which is a substantial improvement.

### 摘要中文翻译

近年来，基于生成式召回的推荐系统成为一种有前景的新范式。然而，大多数现代推荐系统仍采用“召回-排序”的策略，生成模型通常只在召回阶段充当候选选择器。本文提出 OneRec，用统一的生成式模型替代级联学习框架。据作者所述，这是首个在真实场景中显著超过当前复杂、精心设计推荐系统的端到端生成式模型。OneRec 包含三个关键部分：第一，encoder-decoder 结构，用于编码用户历史行为序列，并逐步解码用户可能感兴趣的视频；作者采用稀疏 MoE，在不等比例增加计算量的情况下扩大模型容量。第二，session-wise generation，相比传统 next-item prediction，它以更自然、更具上下文连贯性的方式生成一组推荐结果，避免依赖人工规则组合逐点预测。第三，结合 DPO 的 Iterative Preference Alignment，用于提升生成结果质量。由于推荐系统通常每次请求只有一次曝光机会，无法同时获得正负样本，作者设计 reward model 模拟用户生成，并定制采样策略。实验表明，少量 DPO 样本即可对齐用户兴趣偏好并显著提升生成质量。OneRec 已部署在快手主场景，带来 1.6% 的 watch-time 提升。

### 这篇论文大概在解决什么

这篇论文解决的是：生成式推荐能否不只作为召回阶段的候选生成器，而是直接统一召回与排序，端到端生成最终推荐结果。

核心想法是把用户历史行为编码后，用生成式 decoder 一次生成 session 级别的推荐序列，并通过 MoE 扩大容量、通过 DPO 式偏好对齐提升线上推荐质量。

它关注的不是传统 ANN 向量召回，而是把候选 item 的产生、排序偏好、展示序列组织都放入一个生成式推荐框架里。

### 可能需要精读时重点看什么

- 生成目标：item/video ID 是如何被表示和解码的，是否有 semantic ID 或离散 token 设计。
- session-wise generation：如何避免逐点 next-item prediction 带来的序列拼接问题。
- 召回排序一体化：它如何替代传统 retrieve-and-rank pipeline，线上 serving 如何保证延迟。
- MoE 设计：稀疏 MoE 放在 encoder-decoder 哪些位置，计算量和容量如何权衡。
- Iterative Preference Alignment：推荐系统里 DPO 样本如何构造，reward model 如何模拟用户偏好。
- 与你的方向的连接：如果你关心“生成候选集合 + 后续排序”或“多候选生成质量”，这篇是强相关工业标杆。

