## Neural Functional Transformers

### 决策卡片
- 年份：2023
- 引用数：未提供
- 与搜索意图相关性：高 —— 论文核心是用 attention/Transformer 构造满足权重空间置换对称性的 permutation equivariant / invariant neural functional，并明确涉及 permutation invariant latent representations。
- 是否值得进入精读候选：值得。若你关注 Deep Sets、置换不变/等变网络与 Transformer 的结合，这篇很相关，尤其适合看其如何在神经网络权重空间中定义 attention 层。

### 摘要原文

The recent success of neural networks as implicit representation of data has driven growing interest in neural functionals: models that can process other neural networks as input by operating directly over their weight spaces. Nevertheless, constructing expressive and efficient neural functional architectures that can handle high-dimensional weight-space objects remains challenging. This paper uses the attention mechanism to define a novel set of permutation equivariant weight-space layers and composes them into deep equivariant models called neural functional Transformers (NFTs). NFTs respect weight-space permutation symmetries while incorporating the advantages of attention, which have exhibited remarkable success across multiple domains. In experiments processing the weights of feedforward MLPs and CNNs, we find that NFTs match or exceed the performance of prior weight-space methods. We also leverage NFTs to develop Inr2Array, a novel method for computing permutation invariant latent representations from the weights of implicit neural representations (INRs). Our proposed method improves INR classification accuracy by up to $+17\%$ over existing methods. We provide an implementation of our layers at https://github.com/AllanYangZhou/nfn.

### 摘要中文翻译

近年来，神经网络作为数据的隐式表示取得了成功，这推动了人们对 neural functionals 的兴趣：这类模型可以将其他神经网络作为输入，并直接在其权重空间上进行操作。然而，构建既有表达能力又高效、并且能够处理高维权重空间对象的 neural functional 架构仍然具有挑战性。本文使用注意力机制定义了一组新的置换等变权重空间层，并将其组合成深层等变模型，称为 Neural Functional Transformers，简称 NFTs。NFTs 在尊重权重空间置换对称性的同时，也融入了注意力机制的优势；注意力机制已经在多个领域表现出显著成功。在处理前馈 MLP 和 CNN 权重的实验中，作者发现 NFTs 能够达到或超过先前权重空间方法的性能。作者还利用 NFTs 开发了 Inr2Array，这是一种从隐式神经表示，即 INRs，的权重中计算置换不变潜在表示的新方法。所提出的方法相比已有方法最高可将 INR 分类准确率提升 $+17\%$。作者在 https://github.com/AllanYangZhou/nfn 提供了相关层的实现。

### 这篇论文大概在解决什么

这篇论文关注的是“把神经网络的权重本身作为输入”的模型设计问题。由于神经网络隐藏单元、通道等存在置换对称性，同一个函数可能对应多种权重排列，因此模型需要尊重这些 permutation symmetry。

论文提出用 attention/Transformer 风格的机制，在权重空间中构造 permutation equivariant 层，并进一步得到 Neural Functional Transformers。它还讨论了如何从 INR 权重中得到 permutation invariant 的潜在表示。

### 可能需要精读时重点看什么

- 它定义的 weight-space attention 层如何保证 permutation equivariance。
- 与 Deep Sets / Set Transformer / permutation invariant networks 的关系：是一般集合建模，还是专门针对神经网络权重空间的对称性。
- NFTs 如何处理 MLP 与 CNN 权重，输入对象的结构化方式是什么。
- Inr2Array 如何从 INR 权重得到 permutation invariant latent representation。
- 实验中与 prior weight-space methods 的比较是否能支持其架构优势。
