## MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query

### 决策卡片
- 年份：2025
- 引用数：未提供
- 与搜索意图相关性：低 —— 论文关注多语种、交错多条件、多图像语义检索数据集与 MLLM 微调框架，不涉及“从人工构造正样本簇学习 K 个潜在原型、max_k 相似度路由、ANN union retrieval”等核心机制。
- 是否值得进入精读候选：不建议优先精读；可作为“近期 MLLM/多模态语义检索数据集与训练框架”的背景论文粗读。

### 摘要原文

Semantic retrieval is crucial for modern applications yet remains underexplored in current research. Existing datasets are limited to single languages, single images, or singular retrieval conditions, often failing to fully exploit the expressive capacity of visual information as evidenced by maintained performance when images are replaced with captions. However, practical retrieval scenarios frequently involve interleaved multi-condition queries with multiple images. Hence, this paper introduces MERIT, the first multilingual dataset for interleaved multi-condition semantic retrieval, comprising 320,000 queries with 135,000 products in 5 languages, covering 7 distinct product categories. Extensive experiments on MERIT identify existing models's limitation: focusing solely on global semantic information while neglecting specific conditional elements in queries. Consequently, we propose Coral, a novel fine-tuning framework that adapts pre-trained MLLMs by integrating embedding reconstruction to preserve fine-grained conditional elements and contrastive learning to extract comprehensive global semantics. Experiments demonstrate that Coral achieves a 45.9% performance improvement over conventional approaches on MERIT, with strong generalization capabilities validated across 8 established retrieval benchmarks. Collectively, our contributions - a novel dataset, identification of critical limitations in existing approaches, and an innovative fine-tuning framework - establish a foundation for future research in interleaved multi-condition semantic retrieval.

### 摘要中文翻译

语义检索对现代应用至关重要，但当前研究中仍然探索不足。现有数据集通常局限于单一语言、单张图像或单一检索条件，往往未能充分利用视觉信息的表达能力；这一点可从“用图像描述替换图像后性能仍能保持”这一现象中看出。然而，实际检索场景经常涉及包含多张图像的交错式多条件查询。因此，本文提出 MERIT，这是首个面向交错式多条件语义检索的多语种数据集，包含 32 万个查询、13.5 万个商品，覆盖 5 种语言和 7 个不同商品类别。

基于 MERIT 的大量实验发现，现有模型存在局限：它们往往只关注全局语义信息，而忽略查询中的具体条件元素。为此，作者提出 Coral，这是一种新的微调框架，用于适配预训练 MLLM。Coral 结合了嵌入重构以保留细粒度条件元素，以及对比学习以提取全面的全局语义。实验表明，Coral 在 MERIT 上相比传统方法取得了 45.9% 的性能提升，并且在 8 个已有检索基准上验证了较强的泛化能力。总体而言，本文的贡献包括一个新数据集、对现有方法关键局限的识别，以及一种创新微调框架，为未来交错式多条件语义检索研究奠定基础。

### 这篇论文大概在解决什么

这篇论文主要解决的是：现实商品检索中，用户查询可能同时包含多种语言、多张图片和多个条件，但现有语义检索数据集与模型多半只处理单语言、单图像或单条件查询。

它的核心贡献看起来包括：

- 构建 MERIT：一个多语种、交错多条件的商品语义检索数据集；
- 分析现有模型的问题：过度依赖全局语义，忽视查询里的细粒度条件；
- 提出 Coral：一个面向预训练 MLLM 的微调框架，结合 embedding reconstruction 和 contrastive learning 来提升检索表现。

### 可能需要精读时重点看什么

如果后续决定精读，建议重点看：

- MERIT 查询形式：所谓 “interleaved multi-condition query” 是否包含多图、多文本、多约束组合；
- Coral 的训练目标：embedding reconstruction 如何保留细粒度条件，对比学习如何建模全局语义；
- 检索架构：是否只是单向量/全局 embedding 检索，还是有多向量、多条件匹配机制；
- 与 CLIP / MLLM 检索模型的对比设置；
- 是否存在可借鉴的多条件 query encoding 设计。

但就当前搜索意图而言，它不直接匹配“多原型正样本簇检索 / K latent prototypes / max_k routing / ANN union retrieval”方向，更像是相关背景而非核心候选。
