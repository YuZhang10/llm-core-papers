## Universality of Group Convolutional Neural Networks Based on Ridgelet Analysis on Groups

### 决策卡片
- 年份：2022
- 引用数：1
- 与搜索意图相关性：中 —— 摘要明确提到覆盖 permutation-invariant inputs / Deep Sets，但主题是群卷积神经网络的通用逼近理论与 ridgelet 分析，并非 Transformer。
- 是否值得进入精读候选：有条件值得；如果你关注 Deep Sets / permutation invariant neural networks 的理论基础可纳入候选，如果主要找 Transformer 相关工作则优先级较低。

### 摘要原文

We show the universality of depth-2 group convolutional neural networks (GCNNs) in a unified and constructive manner based on the ridgelet theory. Despite widespread use in applications, the approximation property of (G)CNNs has not been well investigated. The universality of (G)CNNs has been shown since the late 2010s. Yet, our understanding on how (G)CNNs represent functions is incomplete because the past universality theorems have been shown in a case-by-case manner by manually/carefully assigning the network parameters depending on the variety of convolution layers, and in an indirect manner by converting/modifying the (G)CNNs into other universal approximators such as invariant polynomials and fully-connected networks. In this study, we formulate a versatile depth-2 continuous GCNN $S[γ]$ as a nonlinear mapping between group representations, and directly obtain an analysis operator, called the ridgelet trasform, that maps a given function $f$ to the network parameter $γ$ so that $S[γ]=f$. The proposed GCNN covers typical GCNNs such as the cyclic convolution on multi-channel images, networks on permutation-invariant inputs (Deep Sets), and $\mathrm{E}(n)$-equivariant networks. The closed-form expression of the ridgelet transform can describe how the network parameters are organized to represent a function. While it has been known only for fully-connected networks, this study is the first to obtain the ridgelet transform for GCNNs. By discretizing the closed-form expression, we can systematically generate a constructive proof of the $cc$-universality of finite GCNNs. In other words, our universality proofs are more unified and constructive than previous proofs.

### 摘要中文翻译

本文基于 ridgelet 理论，以统一且构造性的方式证明了二层群卷积神经网络 GCNN 的通用性。尽管卷积神经网络和群卷积神经网络在应用中被广泛使用，但它们的逼近性质尚未得到充分研究。自 2010 年代末以来，已有一些关于 GCNN 通用性的结果。然而，我们对于 GCNN 如何表示函数的理解仍不完整，因为以往的通用性定理通常是针对不同卷积层逐案证明，需要手动、仔细地设置网络参数；或者通过把 GCNN 转换、修改为其他已知的通用逼近器，例如不变多项式或全连接网络，从而间接证明。

在这项研究中，作者将一个通用的二层连续 GCNN $S[γ]$ 表述为群表示之间的非线性映射，并直接得到一个称为 ridgelet transform 的分析算子。该算子可以把给定函数 $f$ 映射到网络参数 $γ$，使得 $S[γ]=f$。所提出的 GCNN 覆盖了典型 GCNN，例如多通道图像上的循环卷积、作用于置换不变输入的网络，即 Deep Sets，以及 $\mathrm{E}(n)$-等变网络。ridgelet transform 的闭式表达能够描述网络参数如何组织起来以表示一个函数。此前 ridgelet transform 仅在全连接网络中为人所知，而本文首次为 GCNN 得到了 ridgelet transform。通过离散化该闭式表达，作者可以系统地产生有限 GCNN 的 $cc$-通用性的构造性证明。换言之，本文的通用性证明相比以往更加统一且更具构造性。

### 这篇论文大概在解决什么

这篇论文主要讨论 **群卷积神经网络 GCNN 的通用逼近能力**，目标是用 ridgelet 分析给出一种更统一、更构造性的理论证明框架。

和你的搜索意图的交集在于：摘要中明确说该框架覆盖 **permutation-invariant inputs / Deep Sets**，因此它可能有助于理解 Deep Sets 这类置换不变网络在群表示视角下的通用性。

但它不是一篇以 Transformer 为核心的论文，也不是直接讨论 Set Transformer 或 permutation-invariant Transformer 架构的工作。

### 可能需要精读时重点看什么

- 它如何把 **Deep Sets / permutation-invariant inputs** 纳入 GCNN 框架。
- 通用性证明是否能对应到你关心的置换不变神经网络形式。
- ridgelet transform 在这里是否只是理论工具，还是能给出可操作的参数构造理解。
- 如果你关注 Transformer，需要确认文中是否有任何与 attention / transformer-like set models 的联系；仅从摘要看，这不是重点。
