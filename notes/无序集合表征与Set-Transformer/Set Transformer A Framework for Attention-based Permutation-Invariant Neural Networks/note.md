## Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks

### 一句话定位

Set Transformer 是一个面向无序集合输入的 attention 架构：它把 Transformer 的 self-attention 改造成不依赖输入顺序的集合建模模块，用于学习 permutation-invariant 或 permutation-equivariant 的 set functions。

它的核心价值不是简单“对集合做 pooling”，而是让集合中的元素彼此交互，从而建模 pairwise / higher-order relations。

### 基本信息

- 论文：**Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks**
- 作者：Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh
- 会议：ICML 2019
- arXiv：1810.00825
- 关键词：
  - Set representation
  - Permutation invariance
  - Permutation equivariance
  - Self-attention
  - Inducing points
  - Attention pooling

### 研究问题

很多任务的输入天然是集合，而不是序列：

- 多实例学习：一个 bag 里有多个 instance。
- 点云识别：一个物体由一组 3D points 表示。
- few-shot learning：support set 是一组样本。
- 聚类：输入是一组点，输出若干 cluster parameters。
- 异常检测：给定一个 set，找出其中和其他元素不一致的元素。

集合输入有两个基本要求：

1. **Permutation invariance**
   - 输入元素换顺序，整体输出不应改变。

2. **Variable-size input**
   - 集合大小可以变化。

Deep Sets 的经典形式是：

$$
f(\{x_1, \dots, x_n\}) = \rho\left(\sum_i \phi(x_i)\right)
$$

这类模型满足 permutation invariance，也有通用近似理论，但它有一个明显限制：每个元素先被独立编码，元素间关系主要靠最后 pooling 后的全局向量表达。

如果任务需要比较元素之间的关系，例如“哪些点属于同一簇”“哪张图和其他图不一样”“集合中有几个不同类别”，简单 pooling 往往不够。

### 核心思想

Set Transformer 的基本想法是：

> 对集合做 self-attention，但不加 positional encoding。

Transformer 原本处理序列，位置编码告诉模型 token 的顺序。Set Transformer 面向集合，所以去掉位置编码，让模型只根据元素内容和元素间关系来计算 attention。

它的结构可以分成两部分：

1. **Encoder**
   - 用 SAB 或 ISAB 对集合元素做上下文编码。
   - 输出仍是一个 set，大小通常不变。

2. **Decoder**
   - 用 PMA 把 set 聚合成一个或多个输出。
   - 对整体分类、回归、聚类参数预测等任务都适用。

### 关键模块

#### 1. MAB: Multihead Attention Block

MAB 是基本 attention block：

$$
\text{MAB}(X, Y)
$$

其中 $X$ 是 query，$Y$ 是 key/value。它基本对应 Transformer block，但没有 positional encoding。

直觉：

- $X$ 中每个元素去查看 $Y$ 中哪些元素与自己相关。
- 输出仍然对应 $X$ 的每个元素。

#### 2. SAB: Set Attention Block

SAB 定义为：

$$
\text{SAB}(X) = \text{MAB}(X, X)
$$

也就是集合内部的 self-attention。

输入：

$$
X \in \mathbb{R}^{n \times d}
$$

输出：

$$
Z \in \mathbb{R}^{n \times d}
$$

SAB 是 permutation equivariant 的：

$$
\text{SAB}(\pi X) = \pi \text{SAB}(X)
$$

也就是说，输入元素换顺序，输出元素也只是按同样方式换顺序，而不是变成另一个函数。

这使得 SAB 可以用于元素级任务，例如 set anomaly detection、point-wise labeling、set 内成员打分。

#### 3. ISAB: Induced Set Attention Block

SAB 的复杂度是：

$$
O(n^2)
$$

当集合很大时成本较高。ISAB 引入 $m$ 个可学习的 inducing points：

$$
I \in \mathbb{R}^{m \times d}
$$

计算方式：

$$
H = \text{MAB}(I, X)
$$

$$
\text{ISAB}_m(X) = \text{MAB}(X, H)
$$

直觉上分两步：

1. 少量 inducing points 从整个输入集合中提取全局信息。
2. 原始元素再 attend 到这些全局槽位。

复杂度变成：

$$
O(nm)
$$

其中 $m$ 是固定超参，通常远小于 $n$。

这有点像低秩近似、记忆槽、或 attention bottleneck。

#### 4. PMA: Pooling by Multihead Attention

传统 set pooling 用 mean / sum / max。Set Transformer 用可学习 seed vectors 做 attention pooling：

$$
\text{PMA}_k(Z) = \text{MAB}(S, \text{rFF}(Z))
$$

其中：

- $Z$ 是 encoder 输出的 set features。
- $S \in \mathbb{R}^{k \times d}$ 是 $k$ 个可学习 seed vectors。
- 输出是 $k$ 个向量。

如果 $k=1$：

- 得到一个全局 set representation。
- 可用于整体分类或回归。

如果 $k>1$：

- 得到多个输出向量。
- 可用于聚类中心预测、多目标输出、多兴趣表示等。

论文还会在 PMA 后接 SAB，让多个输出之间继续交互：

$$
H = \text{SAB}(\text{PMA}_k(Z))
$$

这对 amortized clustering 很重要，因为多个 cluster center 之间不是独立的，需要避免多个中心解释同一批点。

### 整体架构

典型 Set Transformer：

$$
\text{Encoder}(X) = \text{SAB}(\text{SAB}(X))
$$

或：

$$
\text{Encoder}(X) = \text{ISAB}_m(\text{ISAB}_m(X))
$$

Decoder：

$$
\text{Decoder}(Z) = \text{rFF}(\text{SAB}(\text{PMA}_k(Z)))
$$

如果是整体 set-level 输出：

- 通常用 $k=1$。

如果是多个相关输出：

- 用 $k>1$，例如聚类中的多个 Gaussian components。

### 理论性质

论文强调两个性质：

1. **Permutation invariance**
   - SAB / ISAB 是 permutation equivariant。
   - PMA 是 permutation invariant。
   - 因此整体 Set Transformer 可以构成 permutation-invariant set function。

2. **Universal approximation**
   - Set Transformer 可以作为 permutation-invariant functions 的 universal approximator。
   - 这继承并扩展了 Deep Sets 的表达能力讨论。

### 实验设计与结论

论文实验的目的不是只证明模型精度高，而是分别验证几个设计点：

- attention pooling 是否比 mean/sum pooling 灵活；
- self-attention encoder 是否能建模元素间交互；
- inducing points 是否能降低复杂度并保持效果；
- PMA 对多个相关输出是否有帮助。

#### 5.1 Maximum Value Regression

任务：输入一组数字，输出最大值。

这个任务很简单，但很能说明 pooling 的差异：

- mean pooling / sum pooling 很难直接恢复最大值。
- max pooling 天然适合。
- PMA 可以学习关注最大元素，因此效果接近 max pooling。

结论：attention pooling 可以根据任务自适应地选择重要元素。

#### 5.2 Counting Unique Characters

任务：从 Omniglot 采样一组手写字符图片，预测集合中有多少个不同字符。

这个任务需要比较图片之间是否属于同一字符类别，而不是独立看每张图。

结果：

- rFF + pooling 表现较弱。
- SAB + PMA 明显更好。
- ISAB 的效果随 inducing points 数量增加而提升。

结论：self-attention encoder 对建模 set 内元素关系有帮助。

#### 5.3 Amortized Clustering with Mixture of Gaussians

任务：输入一组点，直接预测 Gaussian mixture 的参数，而不是运行 EM。

输入：

$$
X = \{x_1, \dots, x_n\}
$$

输出：

$$
\{\pi_j, \mu_j, \sigma_j\}_{j=1}^{k}
$$

这里使用 $k$ 个 PMA seed vectors，每个 seed 负责一个 cluster component。

这个任务特别体现 PMA + SAB 的价值：

- PMA 产生多个输出。
- SAB 让多个输出之间交互，避免多个 cluster center 解释同一区域。

论文发现 ISAB + PMA 效果很好，有时甚至优于 full SAB + PMA。作者认为 inducing points 可能起到 regularization 和 global structure transfer 的作用。

#### 5.4 Set Anomaly Detection

这是一个容易误解但很有意思的实验。

任务不是普通图像分类，而是：

> 给定一个 set，找出其中哪个元素相对于这个 set 是异常的。

数据来自 CelebA。每个小集合包含 8 张人脸：

- 7 张 normal images。
- 1 张 anomaly image。

构造方式：

1. 从 CelebA 的 40 个属性中随机选两个属性，例如：

   ```text
   Black hair & Goatee
   ```

2. 采样 7 张同时具有这两个属性的人脸，作为 normal。

3. 采样 1 张两个属性都不具备的人脸，作为 anomaly。

4. 打乱顺序后输入模型。

所以每个训练样本是：

$$
X = \{x_1, x_2, \dots, x_8\}
$$

标签不是固定类别，而是哪个位置的元素是 anomaly：

$$
y \in \{1, \dots, 8\}
$$

关键点：

> anomaly 是相对于当前 set 定义的，不是某张图片的绝对属性。

同一张图片在一个集合中可能是 normal，在另一个集合中可能是 anomaly。

例如一张“戴眼镜 + 有胡子”的人脸：

- 如果当前 set 的共性是“戴眼镜 + 有胡子”，它就是 normal。
- 如果当前 set 的共性是“黑发 + 山羊胡”，它可能就是 anomaly。

这也是为什么 Set Transformer 适合这个任务：它可以让每个元素在集合上下文中重新编码。

一种自然的模型形式是：

$$
Z = \text{Encoder}(X)
$$

其中：

$$
Z = \{z_1, z_2, \dots, z_8\}
$$

每个 $z_i$ 不再只是第 $i$ 张图片自己的特征，而是：

> 第 $i$ 张图片和整个集合比较之后的上下文表示。

然后对每个 $z_i$ 做 row-wise classifier：

$$
s_i = g(z_i)
$$

再对 8 个 score 做 softmax：

$$
p_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}
$$

训练目标是让真正 anomaly 的位置得分最高。

因此，这个实验不是 set-level classification，而是 set-conditioned element-level classification。

#### 为什么“提取共性”的模型还能做分类？

这里的“分类”不是普通分类：

```text
image -> label
```

而是：

```text
{image_1, ..., image_8} -> 每个 image 的 anomaly score
```

Set Transformer 的 self-attention 做了两件事：

1. 隐式建模当前 set 的 shared pattern。
2. 给每个元素生成相对于这个 shared pattern 的表示。

也就是说，找异常恰恰需要先知道共性是什么。

可以理解为：

```text
先看大家共同有什么特征，
再判断每个元素是否符合这个共同模式。
```

但模型不一定显式生成一个“共性向量”，而是在 attention 过程中让每个元素直接 attend 到其他元素，从而学到“我和大家是否一致”。

#### 5.5 Point Cloud Classification

任务：ModelNet40 点云分类。

输入是一组 3D points：

$$
X \in \mathbb{R}^{n \times 3}
$$

由于点数可能很大，例如 1000 或 5000，full SAB 的 $O(n^2)$ 成本太高，所以实验主要使用 ISAB。

结果显示：

- 小点数时，Set Transformer 有优势。
- 大点数时，ISAB + simple pooling 反而可能更好。

论文解释是：当点数很多时，分类所需信息已经足够丰富，复杂交互的边际收益下降。

### 和 Deep Sets / PointNet 的关系

Deep Sets / PointNet 的基本形式是：

$$
\rho(\text{pool}(\{\phi(x_i)\}))
$$

它们的强点是简单、高效、天然 permutation invariant。

Set Transformer 的区别是：

1. 编码阶段不是独立处理每个元素，而是用 self-attention 让元素之间交互。
2. 聚合阶段不是固定 mean/sum/max，而是用可学习 PMA。
3. 支持多个相关输出，例如多个聚类中心或多个兴趣向量。

可以把 Set Transformer 看作：

> 用 attention 替换 Deep Sets 中的独立元素编码和固定 pooling。

### 和推荐 / 召回的联系

如果把用户历史行为看作一个集合：

$$
\{item_1, item_2, \dots, item_n\}
$$

Set Transformer 可以用来做用户兴趣表征。

潜在优势：

- 建模 item-item 关系。
- 识别重复、互补、主题簇。
- 用 PMA 的多个 seed vectors 抽取多个兴趣向量。
- 对候选召回中的多兴趣建模有启发。

但需要注意：

- 原版 Set Transformer 不建模顺序。
- 如果用户行为序列的时间顺序很重要，直接去掉位置会丢失短期兴趣和转移关系。
- 推荐场景通常需要加入时间、行为类型、位置、session 等特征，或者把 Set Transformer 和序列模型结合。

### 我的理解

这篇论文最值得记住的不是“Transformer 可以处理 set”，而是下面这个拆分：

1. **集合元素怎么互相看见？**
   - SAB / ISAB 解决。

2. **集合怎么聚合成输出？**
   - PMA 解决。

3. **多个输出之间怎么避免互相冲突？**
   - PMA 后接 SAB 解决。

Deep Sets 已经说明了 permutation-invariant function 可以用 pooling 架构近似，但 Set Transformer 更强调实际学习中的 inductive bias：许多 set 任务的难点不是“把元素加起来”，而是“元素之间是什么关系”。

### 局限

1. **SAB 复杂度较高**
   - full self-attention 是 $O(n^2)$。
   - 大集合需要 ISAB 或其他近似。

2. **ISAB 的 inducing points 数量需要调**
   - $m$ 太小可能成为瓶颈。
   - $m$ 太大则计算成本上升。

3. **不适合强顺序任务的原始形式**
   - 如果输入实际是序列，不能简单去掉位置编码。

4. **可解释性有限**
   - attention 可以辅助分析，但不能直接等同于明确规则或因果解释。

### 适合引用的场景

- 需要处理无序集合输入。
- 希望模型满足 permutation invariance / equivariance。
- 简单 pooling 无法建模元素间关系。
- 输入集合大小变化。
- 需要输出多个相关结果，例如多兴趣、多聚类中心、多实例选择。

### 参考链接

- PMLR: https://proceedings.mlr.press/v97/lee19d.html
- PDF: https://proceedings.mlr.press/v97/lee19d/lee19d.pdf
- arXiv: https://arxiv.org/abs/1810.00825
