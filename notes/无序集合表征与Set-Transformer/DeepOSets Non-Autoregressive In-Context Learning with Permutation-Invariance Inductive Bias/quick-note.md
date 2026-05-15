## DeepOSets: Non-Autoregressive In-Context Learning with Permutation-Invariance Inductive Bias

### 决策卡片
- 年份：2024
- 引用数：未提供
- 与搜索意图相关性：高 —— 论文直接涉及 DeepSets、permutation-invariance、Set Transformer，并讨论一种非自回归的 in-context learning 架构与 transformer 替代方案的关系。
- 是否值得进入精读候选：值得，尤其适合关注“集合置换不变神经网络如何替代或补充 Transformer 做 ICL / 回归学习”的方向。

### 摘要原文

In-context learning (ICL) is the remarkable ability displayed by some machine learning models to learn from examples provided in a user prompt without any model parameter updates. ICL was first observed in the domain of large language models, and it has been widely assumed that it is a product of the attention mechanism in autoregressive transformers. In this paper, using stylized regression learning tasks, we demonstrate that ICL can emerge in a non-autoregressive neural architecture with a hard-coded permutation-invariance inductive bias. This novel architecture, called DeepOSets, combines the set learning properties of the DeepSets architecture with the operator learning capabilities of Deep Operator Networks (DeepONets). We provide a representation theorem for permutation-invariant regression learning operators and prove that DeepOSets are universal approximators of this class of operators. We performed comprehensive numerical experiments to evaluate the capabilities of DeepOSets in learning linear, polynomial, and shallow neural network regression, under varying noise levels, dimensionalities, and sample sizes. In the high-dimensional regime, accuracy was enhanced by replacing the DeepSets layer with a Set Transformer. Our results show that DeepOSets deliver accurate and fast results with an order of magnitude fewer parameters than a comparable transformer-based alternative.

### 摘要中文翻译

上下文学习（in-context learning, ICL）是一些机器学习模型展现出的显著能力：它们可以从用户提示中提供的样例进行学习，而不需要更新模型参数。ICL 最早在大语言模型领域被观察到，并且人们普遍认为它是自回归 Transformer 中注意力机制的产物。本文使用风格化的回归学习任务，展示了 ICL 也可以出现在一种非自回归神经网络架构中，该架构内置了置换不变性的归纳偏置。这个新架构称为 DeepOSets，它结合了 DeepSets 的集合学习性质和 Deep Operator Networks（DeepONets）的算子学习能力。作者提出了针对置换不变回归学习算子的表示定理，并证明 DeepOSets 是这类算子的通用逼近器。作者进行了全面的数值实验，评估 DeepOSets 在线性回归、多项式回归和浅层神经网络回归中的学习能力，实验覆盖不同噪声水平、维度和样本量。在高维情形下，通过用 Set Transformer 替换 DeepSets 层可以提升准确率。结果表明，与可比的基于 Transformer 的替代模型相比，DeepOSets 能以少一个数量级的参数实现准确且快速的结果。

### 这篇论文大概在解决什么

这篇论文试图回答：**ICL 是否一定依赖自回归 Transformer 的注意力机制？**  
作者提出 DeepOSets，将 DeepSets 的置换不变集合建模能力与 DeepONets 的算子学习能力结合，用于从一组上下文样例中学习回归任务。核心卖点是：在输入样例天然是集合、顺序不重要的场景中，显式加入 permutation-invariance inductive bias，可能比标准 Transformer 更参数高效。

### 可能需要精读时重点看什么

- DeepOSets 的具体架构：DeepSets 与 DeepONets 是如何组合的。
- 置换不变回归学习算子的表示定理和通用逼近证明。
- 它与标准 Transformer / Set Transformer 的实验对比是否公平。
- 高维场景中为什么 Set Transformer 替换 DeepSets 层会提升表现。
- 任务是否只限于 stylized regression，结论能否推广到更真实的 ICL 场景。
