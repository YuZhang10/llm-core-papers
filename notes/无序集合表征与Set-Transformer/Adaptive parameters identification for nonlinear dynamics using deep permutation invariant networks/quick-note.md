## Adaptive parameters identification for nonlinear dynamics using deep permutation invariant networks

### 决策卡片
- 年份：2025
- 引用数：2（OpenAlex）
- 与搜索意图相关性：高 —— 摘要明确涉及 Deep Sets / Deep Set 与 Set Transformer，并将其用于 permutation-invariant set encoding 的动态系统参数识别。
- 是否值得进入精读候选：值得。若你的重点是 Deep Sets、Set Transformer 在非标准任务中的 permutation-invariant 表征应用，这篇很相关；若只关注理论性质或通用 Transformer 架构，则优先级可稍降。

### 摘要原文

The promising outcomes of dynamical system identification techniques, such as SINDy [Brunton et al. 2016], highlight their advantages in providing qualitative interpretability and extrapolation compared to non-interpretable deep neural networks [Rudin 2019]. These techniques suffer from parameter updating in real-time use cases, especially when the system parameters are likely to change during or between processes. Recently, the OASIS [Bhadriraju et al. 2020] framework introduced a data-driven technique to address the limitations of real-time dynamical system parameters updating, yielding interesting results. Nevertheless, we show in this work that superior performance can be achieved using more advanced model architectures. We present an innovative encoding approach, based mainly on the use of Set Encoding methods of sequence data, which give accurate adaptive model identification for complex dynamic systems, with variable input time series length. Two Set Encoding methods are used, the first is Deep Set [Zaheer et al. 2017], and the second is Set Transformer [Lee et al. 2019]. Comparing Set Transformer to OASIS framework on Lotka Volterra for real-time local dynamical system identification and time series forecasting, we find that the Set Transformer architecture is well adapted to learning relationships within data sets. We then compare the two Set Encoding methods based on the Lorenz system for online global dynamical system identification. Finally, we trained a Deep Set model to perform identification and characterization of abnormalities for 1D heat-transfer problem.

### 摘要中文翻译

动力系统识别技术，例如 SINDy [Brunton et al. 2016]，已经展现出有前景的结果。相比不可解释的深度神经网络 [Rudin 2019]，这类方法在定性可解释性和外推能力方面具有优势。然而，在实时应用场景中，这些技术面临参数更新困难，尤其是当系统参数可能在过程之中或过程之间发生变化时。近期，OASIS [Bhadriraju et al. 2020] 框架提出了一种数据驱动技术，用于解决实时动力系统参数更新的局限，并取得了有趣结果。尽管如此，本文表明，使用更先进的模型架构可以获得更优性能。

作者提出了一种新的编码方法，主要基于对序列数据使用集合编码方法，从而在输入时间序列长度可变的情况下，实现复杂动力系统的精确自适应模型识别。文中使用了两种集合编码方法：第一种是 Deep Set [Zaheer et al. 2017]，第二种是 Set Transformer [Lee et al. 2019]。作者在 Lotka-Volterra 系统上将 Set Transformer 与 OASIS 框架进行比较，用于实时局部动力系统识别和时间序列预测，发现 Set Transformer 架构很适合学习数据集合内部的关系。随后，作者基于 Lorenz 系统比较了两种集合编码方法，用于在线全局动力系统识别。最后，作者训练了一个 Deep Set 模型，用于一维传热问题中的异常识别与表征。

### 这篇论文大概在解决什么

这篇论文关注的是：当非线性动力系统的参数会随时间或工况变化时，如何用 permutation-invariant / set encoding 神经网络对系统参数进行在线或自适应识别。

核心思路是把可变长度时间序列视作集合或集合式输入，用 Deep Set 和 Set Transformer 这类对排列不敏感的结构进行编码，再用于动力系统识别、预测或异常表征。

### 可能需要精读时重点看什么

- 它如何把时间序列转换成 set encoding 输入：是否真的适合 permutation-invariant 建模，还是牺牲了时序顺序信息。
- Deep Set 与 Set Transformer 的具体架构差异，以及各自在动力系统识别中的表现。
- Set Transformer 相比 OASIS 的实验设置是否公平，尤其是在 Lotka-Volterra 任务上的实时识别与预测。
- Lorenz 系统实验中，Deep Set 和 Set Transformer 对在线全局识别的差别。
- 论文是否讨论 variable input time series length，这是与 Deep Sets / Set Transformer 搜索意图比较相关的应用点。
