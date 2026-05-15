## PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets

### 决策卡片
- 年份：2025
- 引用数：18（OpenAlex）
- 与搜索意图相关性：中 —— 论文涉及 PointNet 对点集的置换不变建模，并明确使用对称函数保证 permutation invariance，但核心不是 Deep Sets 理论或 Transformer，而是将 KAN 替代 MLP 用于点云任务。
- 是否值得进入精读候选：可作为候选；如果你关注“置换不变神经网络在点集/点云中的具体架构实现”，值得粗读或选择性精读；如果重点是 Deep Sets 理论或 Set Transformer，则优先级不高。

### 摘要原文

Kolmogorov-Arnold Networks (KANs) have recently gained attention as an alternative to traditional Multilayer Perceptrons (MLPs) in deep learning frameworks. KANs have been integrated into various deep learning architectures such as convolutional neural networks, graph neural networks, and transformers, with their performance evaluated. However, their effectiveness within point-cloud-based neural networks remains unexplored. To address this gap, we incorporate KANs into PointNet for the first time to evaluate their performance on 3D point cloud classification and segmentation tasks. Specifically, we introduce PointNet-KAN, built upon two key components. First, it employs KANs instead of traditional MLPs. Second, it retains the core principle of PointNet by using shared KAN layers and applying symmetric functions for global feature extraction, ensuring permutation invariance with respect to the input features. In traditional MLPs, the goal is to train the weights and biases with fixed activation functions; however, in KANs, the goal is to train the activation functions themselves. We use Jacobi polynomials to construct the KAN layers. We extensively and systematically evaluate PointNet-KAN across various polynomial degrees and special types such as the Lagrange, Chebyshev, and Gegenbauer polynomials. Our results show that PointNet-KAN achieves competitive performance compared to PointNet with MLPs on benchmark datasets for 3D object classification and part and semantic segmentation, despite employing a shallower and simpler network architecture. We also study a hybrid PointNet model incorporating both KAN and MLP layers. We hope this work serves as a foundation and provides guidance for integrating KANs, as an alternative to MLPs, into more advanced point cloud processing architectures.

### 摘要中文翻译

Kolmogorov-Arnold Networks（KAN）最近作为传统多层感知机（MLP）的替代方案，在深度学习框架中受到关注。KAN 已被集成到多种深度学习架构中，例如卷积神经网络、图神经网络和 Transformer，并对其性能进行了评估。然而，KAN 在基于点云的神经网络中的有效性仍未被探索。为填补这一空白，作者首次将 KAN 引入 PointNet，用于评估其在 3D 点云分类和分割任务中的表现。

具体而言，作者提出了 PointNet-KAN，它基于两个关键组成部分。第一，它使用 KAN 替代传统 MLP。第二，它保留了 PointNet 的核心原则：使用共享 KAN 层，并应用对称函数进行全局特征提取，从而保证对输入特征的置换不变性。在传统 MLP 中，训练目标是学习权重和偏置，同时激活函数固定；而在 KAN 中，训练目标是学习激活函数本身。作者使用 Jacobi 多项式构造 KAN 层，并在不同多项式阶数以及 Lagrange、Chebyshev、Gegenbauer 等特殊多项式类型上，对 PointNet-KAN 进行了广泛且系统的评估。

结果显示，尽管 PointNet-KAN 使用了更浅、更简单的网络结构，但它在 3D 物体分类、部件分割和语义分割基准数据集上，相比基于 MLP 的 PointNet 取得了有竞争力的表现。作者还研究了一种同时包含 KAN 和 MLP 层的混合 PointNet 模型。作者希望这项工作能为将 KAN 作为 MLP 替代方案集成到更先进的点云处理架构中提供基础和指导。

### 这篇论文大概在解决什么

这篇论文主要想回答：在 PointNet 这类处理无序点集的点云网络中，能否用 KAN 替代传统 MLP，并在分类、部件分割、语义分割任务上保持或提升效果。

它与搜索意图中的“permutation invariant neural networks”有直接关联，因为 PointNet 本身依赖共享网络层和对称聚合函数来处理无序点集；摘要也明确提到通过 symmetric functions 保证 permutation invariance。不过，它不是专门讨论 Deep Sets 框架，也不是以 Transformer / Set Transformer 为核心。

### 可能需要精读时重点看什么

- PointNet-KAN 如何替代 PointNet 中的 MLP：替换的是哪些层，是否保持共享权重结构。
- 它如何实现点集的置换不变性：尤其是 shared KAN layers + symmetric functions 的设计。
- KAN 相比 MLP 的实际收益：性能、参数量、深度、训练稳定性和计算成本。
- 多项式类型与阶数的消融实验：Jacobi、Lagrange、Chebyshev、Gegenbauer 等选择是否显著影响结果。
- 混合 KAN+MLP 模型是否比纯 KAN 或纯 MLP 更有优势。
- 如果你的重点是 Deep Sets / Set Transformer：可重点查它是否讨论与 PointNet、Deep Sets 或 Transformer 类集合模型的关系；若没有，则不必深读全文。
