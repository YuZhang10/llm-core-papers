## Generative External Knowledge for Zero-shot Action Recognition

### 决策卡片
- 年份：2025
- 引用数：无数据
- 与搜索意图相关性：中 —— 论文涉及 CLIP/Transformer 与 multi-prototype prediction，但场景是零样本动作识别分类，不是面向人工构造正样本簇的多原型检索、max-k 路由或 ANN union retrieval。
- 是否值得进入精读候选：可作为弱相关候选；如果你关注“多原型如何生成/匹配”可以略读方法部分，否则优先级不高。

### 摘要原文

Zero-Shot Action Recognition (ZSAR) aims to infer new action classes without any samples of those classes. Traditional methods of acquiring language and visual knowledge limit the relevance and richness of such knowledge, complicating the generalization to the recognition of unseen classes. To address this issue, this paper proposes a novel method based on generative external knowledge for ZSAR. The method advances an effective multi-prototype prediction strategy for zero-shot action recognition, and significantly improves the ability to accurately match the correct class prototypes. This paper develops video and class processing pipelines, within which an autonomous CLIP encoder first captures modality-specific information, followed by an integrated Transformer that facilitates the learning of cross-modal representations. These pipelines generate distinct cross-modal representations for both videos and classes. For predictions, a multi-prototype method is proposed to classify a video sample into the class corresponding to the most likely prototype. This method uses multiple prototypes per class based on the number of class images. Experiments conducted on three datasets demonstrate substantial improvements over existing ZSAR techniques, verifying the critical roles of generative external knowledge, sophisticated cross-modal fusion, and the multi-prototype strategy.

### 摘要中文翻译

零样本动作识别（ZSAR）旨在在没有某些新动作类别样本的情况下推断这些类别。传统获取语言和视觉知识的方法限制了知识的相关性和丰富性，从而增加了对未见类别进行泛化识别的难度。为了解决这一问题，本文提出了一种基于生成式外部知识的零样本动作识别新方法。该方法提出了一种有效的多原型预测策略，用于零样本动作识别，并显著提升了准确匹配正确类别原型的能力。

本文构建了视频处理和类别处理两条流程，其中自主 CLIP 编码器首先捕获特定模态的信息，随后通过集成 Transformer 学习跨模态表示。这些流程分别为视频和类别生成不同的跨模态表示。在预测阶段，论文提出一种多原型方法，将视频样本分类到最可能原型所对应的类别。该方法根据类别图像数量为每个类别使用多个原型。在三个数据集上的实验表明，该方法相比已有零样本动作识别技术取得了显著提升，验证了生成式外部知识、复杂跨模态融合和多原型策略的关键作用。

### 这篇论文大概在解决什么

这篇论文解决的是零样本动作识别问题：如何在没有目标动作类别训练样本的情况下，利用外部生成知识、CLIP 编码器和 Transformer 跨模态融合，把视频样本匹配到正确的动作类别。

它的关键点包括：

- 使用生成式外部知识增强类别表示；
- 使用 CLIP 提取视觉/语言相关信息；
- 用 Transformer 学习视频与类别之间的跨模态表示；
- 为每个类别构造多个原型，并将视频分配给最可能的类别原型。

### 可能需要精读时重点看什么

- 多原型策略具体如何定义：每个类别的多个 prototype 是如何生成、数量如何确定、是否可学习。
- 匹配机制：是否类似 max-over-prototypes 的相似度路由。
- 类别图像与生成式外部知识的来源：是否可迁移到“人工构造正样本簇”的场景。
- CLIP + Transformer 的跨模态融合方式：是否只是分类任务专用，还是可改造成检索 embedding。
- 与你的搜索意图的差距：论文看起来偏“类别级多原型分类”，未体现 ANN 检索、positive item cluster supervision、latent pattern prototypes 或 union retrieval。
