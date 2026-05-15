## EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration

> 依据说明：本笔记基于题中给出的 arXiv TeX 源码摘录、表格片段和图片解读；部分消融表格不完整，因此消融结论以定性为主。

### 一句话定位

EAGER 是一个用于序列推荐的 **生成式检索 Generative Retrieval** 框架：把“推荐下一个 item”改写成“自回归生成 item 的离散 token identifier”，并用 **behavior stream + semantic stream** 两路生成来协同利用用户行为信息和物品语义信息。

---

### 基本信息

- **论文**：EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration
- **arXiv**：2406.14017
- **会议**：KDD 2024
- **任务**：Sequential Recommendation / Generative Recommendation
- **核心关键词**：
  - Generative Retrieval
  - Autoregressive Generation
  - Semantic Tokenization
  - Behavior-Semantic Collaboration
  - Two-Stream Generation
  - Contrastive Distillation

---

### 摘要中文翻译

生成式检索最近成为序列推荐中的一种有前景方法，它将候选物品召回建模为自回归序列生成问题。现有生成式推荐方法通常只关注物品信息中的行为侧或语义侧，忽略二者的互补性，因此效果受限。

为解决该问题，论文提出 **EAGER**，一个融合行为信息和语义信息的生成式推荐框架。作者指出，联合这两类信息有三个关键挑战：

1. 需要一个能统一处理两种特征类型的生成式架构；
2. 需要保证每类信息都能充分且独立地学习；
3. 需要促进细粒度交互，使两类信息能协同使用。

为此，EAGER 提出：

1. **Two-Stream Generation Architecture**：共享 encoder，两个独立 decoder，分别生成 behavior tokens 和 semantic tokens，并用 confidence-based ranking 合并结果；
2. **Global Contrastive Task**：引入 summary token，使每一路 decoder 获得更强的全局判别能力；
3. **Semantic-guided Transfer Task**：通过重建和识别目标，让语义信息隐式指导行为信息学习。

在四个公开推荐数据集上的实验表明，EAGER 优于已有生成式和传统推荐方法。

---

### 研究问题

传统推荐系统通常是：

1. 学 user/item embedding；
2. 用 ANN index，如 Faiss/ScaNN，做近邻检索。

问题是：**表示学习和索引构建是分离的，难以端到端优化。**

生成式推荐尝试解决这个问题：

- 将每个 item 表示成离散 code sequence：
  \[
  Y = [y_1, y_2, \dots, y_l]
  \]
- 给定用户历史交互序列：
  \[
  X = [x_1, x_2, \dots, x_{t-1}]
  \]
- 模型自回归生成下一个 item 的 code：
  \[
  p(Y|X)=\prod_{i=1}^{l}p(y_i|X,y_1,\dots,y_{i-1})
  \]

但现有方法常有一个基础缺陷：

- RecForest 等偏 **behavior information**；
- TIGER 等偏 **semantic information**；
- 很少系统处理二者如何协作。

EAGER 要解决的基础问题是：

> 在生成式推荐中，如何让 item identifier 同时承载行为偏好和语义先验，并避免直接特征融合带来的冲突？

---

### 核心方法

#### 1. 总体框架

EAGER 包含三部分：

1. **TSG: Two-Stream Generation Architecture**
2. **GCT: Global Contrastive Task**
3. **STT: Semantic-guided Transfer Task**

最终训练目标：

\[
\mathcal{L}_{EAGER}
=
\mathcal{L}_{gen}
+
\lambda_1 \mathcal{L}_{con}
+
\lambda_2(\mathcal{L}_{recon}+\mathcal{L}_{recog})
\]

---

#### 2. Two-Stream Generation Architecture

核心思想：**不要在 encoder 端直接融合 behavior 和 semantic，而是在 decoder 端用两路监督分别生成。**

##### Shared Encoder

用户历史交互序列经过 Transformer encoder：

\[
H = Encoder(X)
\]

这里共享 encoder，而不是两个 encoder。原因是用户历史序列本身是统一输入，后续差异交给两个 decoder 处理。

##### Dual Codes

对每个 item 构造两套离散 code：

- behavior code：
  \[
  Y^b = [y^b_1,\dots,y^b_l]
  \]
- semantic code：
  \[
  Y^s = [y^s_1,\dots,y^s_l]
  \]

来源：

- behavior embedding \(E^b\)：由推荐模型或行为模型提取，例如 DIN；
- semantic embedding \(E^s\)：由语义模型提取，例如 Sentence-T5。

然后分别对两类 embedding 做 **hierarchical k-means**，得到树状离散 token。每个 item 最终对应一条 token 路径。

##### Dual Decoders

两个 decoder 分别生成：

\[
p(Y^b|X), \quad p(Y^s|X)
\]

生成损失：

\[
\mathcal{L}_{gen}
=
\mathcal{L}^b_{gen}
+
\mathcal{L}^s_{gen}
\]

其中：

\[
\mathcal{L}^t_{gen}
=
-\sum_{i=1}^{l}
\log p(y_i^t|H,y^t_{<i}), \quad t\in\{b,s\}
\]

这种设计的意义：

- 避免 behavior/semantic 过早融合；
- 两个 decoder 可分别适配两种 code 分布；
- 推理时两路并行生成，效率高于串行生成两种 identifier。

---

#### 3. Global Contrastive Task

问题：普通自回归生成只逐 token 学局部条件概率，不一定学到 item 的全局判别表示。

EAGER 在 code 末尾加入 summary token：

\[
Y^t = [y_{SOS}^t,y_1^t,\dots,y_l^t,y_{EOS}^t]
\]

其中 \(y_{EOS}^t\) 负责汇总前面 token 的全局信息。

然后用 contrastive distillation 让 summary token 对齐预训练 item embedding：

\[
\mathcal{L}_{con}^t
=
F(y^t_{EOS}, E^t), \quad t\in\{b,s\}
\]

论文中采用 positive-only contrastive metric，例如 Smooth \(L_1\)。

直观理解：

- decoder 不只会“拼出 code”；
- 还要让生成路径形成可判别的 item 表示；
- \(EOS\) 类似 Transformer 里的 `[CLS]`/summary token。

---

#### 4. Semantic-guided Transfer Task

即使两路 decoder 独立学习，仍需要信息交互。但作者认为直接 feature-level fusion 容易冲突，因此设计一个辅助任务，让语义隐式指导行为。

做法：

1. 取 behavior code：
   \[
   [y^b_{CLS}, y^b_1,\dots,y^b_l]
   \]
2. 取 semantic summary token：
   \[
   y^s_{EOS}
   \]
3. 用一个辅助 bidirectional Transformer，通过 cross-attention 让 behavior token attend 到 semantic global feature；
4. 输出：
   \[
   [r_{CLS}, r_1,\dots,r_l]
   \]

包含两个目标。

##### Reconstruction

随机 mask 一些 behavior tokens，让模型在 semantic summary 的帮助下重建它们。

本质是：

> 语义全局信息应该能帮助恢复行为 token。

负采样形式类似 sampled softmax：

\[
\mathcal{L}_{recon}
=
-\sum_i
\log
\frac{
\exp(r_{m_i}^{\top}y_i)
}{
\exp(r_{m_i}^{\top}y_i)
+
\sum_{j=1}^{J}\exp(r_{m_i}^{\top}y_j)
}
\]

##### Recognition

构造正负样本，判断 behavior code 是否与 semantic summary 匹配。

\[
\mathcal{L}_{recog}
=
-\log s^+
-
\log(1-s^-)
\]

其中：

- \(s^+\)：正样本匹配分数；
- \(s^-\)：替换部分 behavior token 后的负样本分数。

这个模块的关键不是直接融合，而是通过任务约束实现 **implicit interaction**。

---

#### 5. 推理机制

推理时：

1. behavior decoder 用 beam search 得到 top-\(k\) item codes；
2. semantic decoder 用 beam search 得到 top-\(k\) item codes；
3. 合并得到 \(2k\) 个候选；
4. 根据生成 log probability / perplexity 类似的 confidence score 排序；
5. 输出最终 top-\(k\)。

低 entropy / 低 perplexity 表示模型更有信心。

---

### 关键图表解读

#### 图：Contrastive Item Embedding vs Generative Item Embedding

图片显示：

- Contrastive item embedding 分布更分散；
- Generative item embedding 更集中，判别性较弱。

含义：

> 纯生成式训练容易让 item 表示压缩在较小区域，缺乏全局区分能力。

这正是 GCT 的动机：用 summary token + contrastive distillation 补足判别性。

---

#### 图：Attention Distance across Transformer Blocks

图中比较 ID 与 Modality 在不同 Transformer block 的 attention distance。

现象：

- ID 和 modality 的 attention pattern 不一致；
- modality 在后层往往呈现更远距离注意；
- ID 行为信息在不同层波动更大。

含义：

> 行为信息和语义信息的建模机制不同，直接混合可能造成表示冲突。

这支持 EAGER 的设计：不在输入侧粗暴融合，而是使用 two-stream decoder 和辅助交互任务。

---

#### 图：Performance Comparison：I / M / I+M

图中比较 DIN、SASRec 使用：

- I：ID / behavior 信息；
- M：modality / semantic 信息；
- I+M：二者直接融合。

观察：

- 在 DIN 上，M 和 I+M 明显优于单独 I；
- 在 SASRec 上，I 反而优于 M 和 I+M；
- 直接 I+M 并不总是提升。

结论：

> behavior 与 semantic 是互补的，但简单 concat / early fusion 不稳定。

EAGER 的 two-stream + late ranking 是对这个问题的结构性处理。

---

#### 图：Beauty / Toys 上不同 code 配置

图中横轴是：

\[
(\#Branch, \#Length)
\]

即 hierarchical k-means 的分支数和 code 长度。

观察：

- 在 Beauty 和 Toys 上，our 始终高于 base；
- 不同 tokenization 配置下提升都存在。

说明：

> EAGER 对离散 code 结构具有一定鲁棒性，不只是某个超参组合偶然有效。

---

#### 图：Layer 数量影响

Beauty 数据集上，随着层数从 1 到 4 增加：

- Recall@10 上升；
- NDCG@10 上升。

说明：

> 更深的 Transformer 有助于建模生成式推荐中的序列依赖和 token 结构。

但图中只到 4 层，不能推出无限加深一定更好。

---

### 关键贡献

1. **提出 behavior-semantic collaboration 的生成式推荐框架**
   - 不是只用行为 token；
   - 也不是只用语义 token；
   - 而是两路生成、协同排序。

2. **Two-stream generation**
   - shared encoder；
   - behavior decoder；
   - semantic decoder；
   - confidence-based ranking 合并结果。

3. **Global contrastive task**
   - 在生成式 decoder 中加入 summary token；
   - 用 contrastive distillation 增强全局判别能力。

4. **Semantic-guided transfer task**
   - 通过 reconstruction 和 recognition 实现语义到行为的隐式知识迁移；
   - 避免直接 feature fusion 的不稳定。

---

### 实验与结论

#### 数据集

四个公开序列推荐数据集：

| Dataset | Users | Items | Interactions | Density |
|---|---:|---:|---:|---:|
| Beauty | 22,363 | 12,101 | 198,360 | 0.00073 |
| Sports and Outdoors | 35,598 | 18,357 | 296,175 | 0.00045 |
| Toys and Games | 19,412 | 11,924 | 167,526 | 0.00073 |
| Yelp | 30,431 | 20,033 | 316,354 | 0.00051 |

#### Baselines

包括：

- Traditional：GRU4REC, Caser, HGN
- Transformer-based：SASRec, BERT4Rec, \(S^3\)-Rec
- Tree-based：TDM, RecForest
- Generative：TIGER

#### 主要结果

EAGER 在多数指标上取得最优。

典型结果：

- **Beauty**
  - Recall@10：EAGER 0.0836，优于最佳 baseline 约 25.90%
  - NDCG@10：EAGER 0.0525，提升约 31.25%

- **Sports**
  - 提升较小但整体稳定；
  - NDCG@20 提升约 11.92%。

- **Toys**
  - NDCG 指标提升明显；
  - 但 Recall@20 为 0.1024，低于 \(S^3\)-Rec 的 0.1065，说明 EAGER 并非所有指标全胜。

- **Yelp**
  - Recall@20：EAGER 0.0724；
  - NDCG@20：EAGER 0.0311；
  - 相比最佳 baseline 有明显提升。

#### 消融结论

根据论文描述和部分表格：

- 去掉 TSG、GCT、STT 都会降低效果；
- TSG 是主结构；
- GCT 增强生成 token 的全局判别能力；
- STT 带来 behavior-semantic 的隐式交互收益。

但题中消融表格截断，无法精确复现完整数值。

---

### 局限性

1. **依赖预训练 embedding**
   - behavior embedding 依赖 DIN 等行为模型；
   - semantic embedding 依赖 Sentence-T5 等语义模型；
   - 上游 encoder 质量会影响 code 质量。

2. **离散 tokenization 是离线过程**
   - hierarchical k-means 不是端到端联合优化；
   - code 结构可能影响最终召回。

3. **两路 decoder 增加计算与存储**
   - 相比单路生成式模型，训练和推理复杂度更高。

4. **confidence-based ranking 需要校准**
   - behavior stream 和 semantic stream 的概率分布未必天然可比；
   - 用 log probability / entropy 合并可能存在 calibration 问题。

5. **语义指导是单向的**
   - STT 主要是 semantic-guided behavior；
   - 没有充分讨论 behavior-guided semantic 是否也有收益。

6. **并非所有指标全胜**
   - 例如 Toys 的 Recall@20 低于 \(S^3\)-Rec。

---

### 放进大模型基础知识体系里怎么理解

这篇论文可以放在三个基础知识点下理解。

#### 1. Generative Retrieval：把检索变成生成

传统检索：

\[
query/user \rightarrow embedding \rightarrow ANN search
\]

生成式检索：

\[
query/user \rightarrow autoregressive decoder \rightarrow item identifier
\]

EAGER 属于后者。它把推荐系统中的 item 看成“可生成的离散语言”。

---

#### 2. Tokenization 不只用于自然语言

LLM 里 token 是文本子词；EAGER 里 token 是 item code。

item code 来自 embedding 聚类：

\[
item \rightarrow embedding \rightarrow hierarchical\ k\text{-}means \rightarrow code sequence
\]

这说明：

> 大模型思想中的“离散 token 序列建模”可以扩展到文档、图片、商品、用户行为等对象。

---

#### 3. Multi-modal / multi-source fusion 的关键是“何时融合”

EAGER 的经验是：

- early fusion：简单，但可能冲突；
- late fusion：更稳定；
- auxiliary task interaction：在不破坏各自分布的前提下共享信息。

这和多模态大模型中的设计问题一致：

> 不同模态/信息源的表征空间不同，融合位置和融合方式会显著影响效果。

---

### 我需要记住什么

1. **EAGER 解决的问题**：生成式推荐不能只看 behavior 或 semantic，需要二者协同。

2. **核心结构**：
   \[
   shared\ encoder + behavior\ decoder + semantic\ decoder
   \]

3. **item 有两套 code**：
   - behavior code：来自用户交互行为；
   - semantic code：来自文本/模态语义。

4. **GCT 的作用**：用 summary token 增强生成式 decoder 的全局判别能力。

5. **STT 的作用**：用语义信息隐式指导行为 token 学习，而不是粗暴融合特征。

6. **推理方式**：两路 beam search，合并 \(2k\) 个候选，用 confidence score 排序。

7. **基础启发**：
   - 推荐可以被建模成 token generation；
   - item identifier 的设计很重要；
   - 多信息源融合不能只靠 concat，结构设计更关键。
