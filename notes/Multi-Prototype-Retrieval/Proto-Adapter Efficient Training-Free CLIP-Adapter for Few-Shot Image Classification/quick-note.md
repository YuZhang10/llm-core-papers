## Proto-Adapter: Efficient Training-Free CLIP-Adapter for Few-Shot Image Classification

### 决策卡片
- 年份：2024
- 引用数：未提供
- 与搜索意图相关性：中：论文与 CLIP、few-shot 原型表示和样本聚合有关，但摘要中是“每类单原型/固定大小 adapter”，不是搜索意图中的 K 个潜在多原型、max-k 路由或 ANN union retrieval。
- 是否值得进入精读候选：可作为“CLIP 原型化适配/样本压缩”的参考候选，但不属于多原型检索主线，建议先略读方法图与 prototype 构造部分。

### 摘要原文

Large vision-language models, such as Contrastive Vision-Language Pre-training (CLIP), pre-trained on large-scale image-text datasets, have demonstrated robust zero-shot transfer capabilities across various downstream tasks. To further enhance the few-shot recognition performance of CLIP, Tip-Adapter augments the CLIP model with an adapter that incorporates a key-value cache model constructed from the few-shot training set. This approach enables training-free adaptation and has shown significant improvements in few-shot recognition, especially with additional fine-tuning. However, the size of the adapter increases in proportion to the number of training samples, making it difficult to deploy in practical applications. In this paper, we propose a novel CLIP adaptation method, named Proto-Adapter, which employs a single-layer adapter of constant size regardless of the amount of training data and even outperforms Tip-Adapter. Proto-Adapter constructs the adapter's weights based on prototype representations for each class. By aggregating the features of the training samples, it successfully reduces the size of the adapter without compromising performance. Moreover, the performance of the model can be further enhanced by fine-tuning the adapter's weights using a distance margin penalty, which imposes additional inter-class discrepancy to the output logits.

### 摘要中文翻译

大型视觉-语言模型，例如在大规模图文数据集上预训练的对比视觉-语言预训练模型 CLIP，已经在多种下游任务中展现出强大的零样本迁移能力。为了进一步提升 CLIP 的少样本识别性能，Tip-Adapter 为 CLIP 增加了一个 adapter，该 adapter 包含由少样本训练集构建的键值缓存模型。这种方法能够实现免训练适配，并且在少样本识别中取得了显著提升，尤其是在额外微调后效果更好。然而，adapter 的大小会随着训练样本数量成比例增加，使其难以在实际应用中部署。本文提出了一种新的 CLIP 适配方法，名为 Proto-Adapter。该方法使用一个单层、固定大小的 adapter，其大小不随训练数据量变化，并且性能甚至超过 Tip-Adapter。Proto-Adapter 基于每个类别的原型表示来构建 adapter 权重。通过聚合训练样本的特征，它在不牺牲性能的情况下成功减小了 adapter 的规模。此外，通过使用距离间隔惩罚来微调 adapter 权重，还可以进一步提升模型性能；该惩罚会在输出 logits 中引入额外的类间差异。

### 这篇论文大概在解决什么

这篇论文主要解决 CLIP 少样本分类适配中的 adapter 规模问题。已有 Tip-Adapter 使用 few-shot 样本构建 key-value cache，样本越多 adapter 越大；Proto-Adapter 则把每个类别的训练样本特征聚合成类别原型，用固定大小的单层 adapter 替代随样本数增长的缓存结构，从而降低部署成本，同时保持或提升分类性能。

它和你的搜索意图有一定交集：都是从正样本/少样本集合中构造 prototype 表示，并用于 CLIP 相关任务。但摘要里看不出它涉及：
- 每个正样本簇学习 K 个 latent pattern prototypes；
- cluster-level supervision；
- max_k similarity routing；
- ANN union retrieval；
- 多原型检索式召回。

因此它更像是“CLIP few-shot 分类中的类原型 adapter 压缩方法”，而不是“多原型检索框架”。

### 可能需要精读时重点看什么

- Proto-Adapter 如何从 few-shot 样本特征构造每类 prototype。
- adapter 权重与 class prototype 的对应关系，是否能迁移到“正样本簇 prototype”场景。
- 它与 Tip-Adapter 的 key-value cache 的差异：是如何从样本级缓存压缩到类别级原型的。
- 是否只用单个 class prototype，还是有可扩展到 multi-prototype 的空间。
- distance margin penalty 的形式：是否对类间分离、prototype 学习有启发。
- 实验中 adapter size、few-shot shot 数变化、性能-存储权衡是否对你的检索系统有参考价值。
