## Set Norm and Equivariant Skip Connections: Putting the Deep in Deep Sets

### 决策卡片
- 年份：2022
- 引用数：6（OpenAlex）
- 与搜索意图相关性：高 —— 论文直接讨论 permutation invariant neural networks，核心对象包括 Deep Sets 和 Set Transformer，并提出改进版 Deep Sets++ / Set Transformer++。
- 是否值得进入精读候选：值得。若你关注 Deep Sets、集合输入的置换不变/等变网络、Set Transformer 的深层训练稳定性，这篇很相关。

### 摘要原文

Permutation invariant neural networks are a promising tool for making predictions from sets. However, we show that existing permutation invariant architectures, Deep Sets and Set Transformer, can suffer from vanishing or exploding gradients when they are deep. Additionally, layer norm, the normalization of choice in Set Transformer, can hurt performance by removing information useful for prediction. To address these issues, we introduce the clean path principle for equivariant residual connections and develop set norm, a normalization tailored for sets. With these, we build Deep Sets++ and Set Transformer++, models that reach high depths with comparable or better performance than their original counterparts on a diverse suite of tasks. We additionally introduce Flow-RBC, a new single-cell dataset and real-world application of permutation invariant prediction. We open-source our data and code here: https://github.com/rajesh-lab/deep_permutation_invariant.

### 摘要中文翻译

置换不变神经网络是从集合数据中进行预测的一类有前景工具。然而，作者指出，现有的置换不变架构，如 Deep Sets 和 Set Transformer，在网络加深时可能会出现梯度消失或梯度爆炸问题。此外，Set Transformer 中常用的 Layer Norm 可能会移除对预测有用的信息，从而损害性能。为了解决这些问题，作者提出了用于等变残差连接的 clean path principle，并设计了专门面向集合数据的 set norm 归一化方法。基于这些设计，作者构建了 Deep Sets++ 和 Set Transformer++，这些模型能够达到更深的层数，并在多种任务上取得与原始模型相当或更好的表现。作者还引入了 Flow-RBC，一个新的单细胞数据集，也是置换不变预测的真实应用场景。数据和代码开源于：https://github.com/rajesh-lab/deep_permutation_invariant。

### 这篇论文大概在解决什么

这篇论文关注的是：如何让 Deep Sets 和 Set Transformer 这类集合建模网络变得更“深”且更稳定。  
摘要显示，作者认为原始架构在加深后会遇到梯度不稳定问题，并且常规 Layer Norm 可能不适合集合任务。因此他们提出了两类改进：等变残差连接原则和 set norm，进而得到 Deep Sets++ 与 Set Transformer++。

### 可能需要精读时重点看什么

- Deep Sets / Set Transformer 深层化时梯度消失或爆炸的具体分析。
- “clean path principle” 如何定义，为什么适合 permutation equivariant residual connections。
- set norm 与 layer norm 的区别，尤其是它如何保留集合预测所需信息。
- Deep Sets++ 和 Set Transformer++ 的架构改动是否容易迁移到你的任务。
- 实验任务是否覆盖你关心的集合输入场景，尤其是集合大小变化、元素交互、单细胞数据等。
