## Recognizing Temporary Construction Site Objects using CLIP-based Few-shot Learning and Multi-modal Prototypes

### 决策卡片
- 年份：2024
- 引用数：未提供
- 与搜索意图相关性：中 —— 论文涉及 CLIP、few-shot、多模态 prototypes，但从摘要看主要是训练-free 的临时施工对象分类，不是面向正样本簇的 K 个潜在原型学习、max-k 路由或 ANN union retrieval。
- 是否值得进入精读候选：可作为次级候选；如果你想找 CLIP 原型构造 / prototype cache / few-shot 多模态分类方法，可以粗读方法部分，否则与“multi-prototype retrieval”主线不完全贴合。

### 摘要原文

Visual understanding of temporary on-site objects is essential for robots and project management in construction. Implementation of deep learning algorithms is challenging on construction sites due to high data annotation cost, demanding computational power, and lack of large-scale training datasets. Recognizing on-site temporary objects demands the algorithms to learn in a data-efficient way. This paper developed a training-free CLIP-based few-shot object classification algorithm with multi-modal prototypes, introduced an ImageNet-based Similarity Cache with image-text similarity features, and achieved state-of-the-art performance on the public dataset SODA and the proposed dataset TOCS.

### 摘要中文翻译

施工现场临时物体的视觉理解对于机器人和项目管理至关重要。由于数据标注成本高、计算资源需求大，以及缺乏大规模训练数据集，在施工现场部署深度学习算法具有挑战性。识别现场临时物体要求算法能够以数据高效的方式学习。本文提出了一种基于 CLIP 的免训练 few-shot 物体分类算法，使用多模态原型；同时引入了一个基于 ImageNet 的相似度缓存，包含图像-文本相似度特征，并在公开数据集 SODA 和作者提出的数据集 TOCS 上取得了当前最优性能。

### 这篇论文大概在解决什么

这篇论文面向施工场景中的临时物体识别问题，重点是如何在标注数据少、训练资源有限的情况下进行物体分类。它利用 CLIP 的图文表征能力，通过 few-shot 和多模态 prototypes 来做训练-free 分类，并用相似度缓存增强识别效果。

它更像是一个“CLIP + 原型 + few-shot 分类”的应用型方法，而不是明确的“多原型检索 / 正样本簇建模 / ANN 召回”论文。

### 可能需要精读时重点看什么

- 多模态 prototypes 是如何构造的：图像原型、文本原型，还是图文混合原型。
- ImageNet-based Similarity Cache 的机制：是否类似检索缓存、特征库或 prototype memory。
- 分类时的相似度计算方式：是否有 max over prototypes、prototype ensemble、加权融合等机制。
- 是否支持每类多个 prototype，以及这些 prototype 是否来自 few-shot 样本或文本模板。
- 方法是否可以迁移到你的场景：把“施工物体类别”替换成“人工构造正样本簇”，用 CLIP embedding + 多 prototype 做候选召回。
