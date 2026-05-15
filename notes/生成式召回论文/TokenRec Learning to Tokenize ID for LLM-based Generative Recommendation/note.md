## TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation

### 一句话定位

TokenRec 解决的是 **LLM-based 推荐系统中“用户/物品 ID 如何被离散 token 化，并且同时保留协同过滤 collaborative filtering 信号”** 的基础问题：把用户和物品从传统推荐里的连续 ID embedding，转成 LLM 可处理的离散 token，同时避免自回归生成物品 token 的低效推理。

---

### 基本信息

- **论文**：TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation
- **arXiv**：2406.10450
- **作者**：Haohao Qu, Wenqi Fan, Zihuai Zhao, Qing Li
- **时间**：2024-06-15
- **关键词**：
  - LLM-based Recommendation
  - ID Tokenization
  - Vector Quantization, VQ
  - Collaborative Filtering, CF
  - Generative Retrieval
- **代码**：论文摘要中给出 GitHub：`https://github.com/Quhaoh233/TokenRec`
- **依据说明**：笔记基于给定 TeX 摘录、摘要和输入图片。完整实验表格数值未完整展示，因此实验结论只概括趋势，不写未核验的具体提升幅度。

---

### 摘要中文翻译

近年来，利用大语言模型推动下一代推荐系统受到关注，因为 LLM 具备较强的语言理解、推理和 in-context learning 能力。在这一场景下，如何对用户和物品进行 tokenization / indexing，是让 LLM 与推荐任务对齐的关键。

已有方法尝试用文本内容或潜在表示来表示用户和物品，但仍存在两个问题：

1. 难以把高阶协同过滤知识有效编码进 LLM 兼容的离散 token；
2. 对训练语料中未出现的新用户或新物品泛化能力不足。

为此，论文提出 **TokenRec**。它包含：

1. **Masked Vector-Quantized Tokenizer, MQ-Tokenizer**：  
   对从协同过滤模型中学到的用户/物品表示进行 mask 和 vector quantization，将其量化为离散 token。
2. **Generative Retrieval paradigm**：  
   LLM 不再自回归生成物品 token 序列，而是生成一个推荐表示，再从物品库中检索 top-$K$ 物品，从而减少 beam search 和 autoregressive decoding 的推理开销。

实验表明，TokenRec 优于传统推荐模型和新兴 LLM 推荐模型。

---

### 研究问题

#### 1. 为什么 LLM 推荐需要 ID tokenization？

传统推荐系统中，用户和物品通常是离散 ID：

\[
u_i \rightarrow \mathbf{p}_i,\quad v_j \rightarrow \mathbf{q}_j
\]

其中 $\mathbf{p}_i, \mathbf{q}_j$ 是可学习 embedding。

但 LLM 的输入是 token 序列。于是推荐系统里的用户/物品 ID 必须转成 LLM 可接受的 token。

---

#### 2. 现有 ID tokenization 的问题

论文重点批判了几类方案：

| 方法 | 问题 |
|---|---|
| Independent Indexing, IID | 给每个用户/物品分配专属 token，会导致词表爆炸，真实系统中用户/物品可能是亿级 |
| Textual title indexing | 用标题、描述等文本表示物品，但文本不一定包含协同过滤信号 |
| Whole-word embedding | 可以缓解 subword 问题，但仍难表达高阶用户-物品交互结构 |
| Continuous indexing / soft prompt | 连续向量难以和 LLM 的离散 token 机制自然对齐 |
| 直接自回归生成物品 ID | 推理慢，需要 decoding / beam search，而且生成空间巨大 |

---

#### 3. 论文要解决的基础问题

核心问题可以写成：

> 如何把推荐系统中的用户/物品 ID，转换成 LLM 兼容的少量离散 token，同时让这些 token 携带 collaborative filtering 中的高阶交互知识，并能泛化到新用户/新物品？

---

### 核心方法

TokenRec 有两个主模块：

1. **MQ-Tokenizer**：学习用户/物品 ID 的离散 token 表示；
2. **Generative Retrieval**：用 LLM 生成推荐向量，再做最近邻检索。

---

#### 1. 整体任务形式

传统 CF 目标是根据用户历史交互 $N(u_i)$ 推荐物品。

论文把它改写成 LLM 形式：

\[
\mathbf{z}_i = \text{LLM}(P, T_i, \{T_j \mid v_j \in N(u_i)\})
\]

其中：

- $P$：prompt；
- $T_i$：用户 $u_i$ 的 tokenized ID；
- $T_j$：历史交互物品 $v_j$ 的 tokenized ID；
- $\mathbf{z}_i$：LLM 生成的用户偏好 / 推荐表示。

注意：用户历史物品不是严格按序列推荐建模，而是更接近 CF 的非序列交互集合。

---

#### 2. Collaborative Representation：先从 GNN-CF 学连续表示

TokenRec 不直接从文本开始，而是先用 GNN-based CF 模型学习用户和物品表示：

\[
\mathbf{p}_i \in \mathbb{R}^d,\quad \mathbf{q}_j \in \mathbb{R}^d
\]

这些表示包含高阶协同过滤知识，例如：

- 相似用户喜欢相似物品；
- 物品通过共同用户形成相似性；
- 用户-物品图上的多跳关系。

论文的关键思想是：

> 先让 GNN-CF 捕获协同结构，再把这些连续表示量化成 LLM 可用的离散 token。

---

#### 3. MQ-Tokenizer：Masked Vector-Quantized Tokenizer

MQ-Tokenizer 对用户和物品分别训练。两者结构相同，论文主要讲 item MQ-Tokenizer。

##### 3.1 Masking Operation

对原始 CF 表示做随机 mask：

\[
E \sim \text{Bernoulli}(\rho)
\]

\[
\mathbf{p}'_i = \text{Mask}(\mathbf{p}_i, E)
\]

\[
\mathbf{q}'_j = \text{Mask}(\mathbf{q}_j, E)
\]

其中 $\rho$ 是 masking ratio。

作用：

- 增强 tokenizer 鲁棒性；
- 避免只记住原始 embedding；
- 让 tokenizer 能在信息缺失时仍恢复协同表示；
- 提升新用户/新物品泛化能力。

---

##### 3.2 $K$-way Encoder

TokenRec 不用一个 encoder，而是用 $K$ 个 encoder：

\[
\mathbf{a}^k_j = \text{Enc}^k(\mathbf{q}'_j) = \text{MLP}^k(\mathbf{q}'_j)
\]

其中：

- $k = 1,2,\dots,K$；
- 每个 encoder 负责提取一种 latent pattern；
- $\mathbf{a}^k_j \in \mathbb{R}^{d_c}$。

直观理解：

> 一个物品不是被一个 token 表示，而是被 $K$ 个 codebook token 组合表示，类似 product quantization / multi-codebook representation。

---

##### 3.3 $K$-way Codebook

每个 encoder 对应一个 codebook：

\[
C = \{C^1, C^2, \dots, C^K\}
\]

\[
C^k \in \mathbb{R}^{L \times d_c}
\]

其中：

- $K$：codebook 个数，也就是每个用户/物品用多少个离散 token 表示；
- $L$：每个 codebook 的 codeword 数；
- $C^k_l$：第 $k$ 个 codebook 中第 $l$ 个 codeword embedding。

对每个 encoder 输出做最近邻量化：

\[
w^k_j = \arg\min_l \|\mathbf{a}^k_j - \mathbf{c}^k_l\|_2^2
\]

\[
\text{Quantize}(\mathbf{a}^k_j) = \mathbf{c}^k_{w^k_j}
\]

于是物品 $v_j$ 被表示为：

\[
v_j \rightarrow \{w^1_j, w^2_j, \dots, w^K_j\}
\]

对应 token embedding 为：

\[
[\mathbf{c}^1_{w^1_j}, \mathbf{c}^2_{w^2_j}, \dots, \mathbf{c}^K_{w^K_j}]
\]

---

##### 3.4 $K$-to-1 Decoder

把 $K$ 个 codeword embedding 聚合后重构原始 CF 表示：

\[
\mathbf{r}_j = \text{Dec}(w^1_j,\dots,w^K_j)
\]

论文采用平均池化加 MLP：

\[
\mathbf{r}_j = \text{MLP}\left(\frac{1}{K}\sum_{k=1}^{K}\mathbf{c}^k_{w^k_j}\right)
\]

目标是让：

\[
\mathbf{r}_j \approx \mathbf{q}_j
\]

也就是离散 token 组合能够保留原始协同表示的信息。

---

##### 3.5 训练目标

主要有重构损失：

\[
\mathcal{L}^{Item}_{recon} = \|\mathbf{q}_j - \mathbf{r}_j\|_2^2
\]

codebook loss：

\[
\mathcal{L}^{Item}_{cb}
=
\sum_{k=1}^{K}
\left\|
\text{sg}[\text{Enc}^k(\mathbf{q}'_j)]
-
\mathbf{c}^k_{w^k_j}
\right\|_2^2
\]

其中 $\text{sg}[\cdot]$ 是 stop-gradient。

由于 $\arg\min$ 不可导，论文使用 **straight-through gradient estimator**，这是 VQ-VAE 系列常见技巧。

给定摘录在 commitment loss 处被截断，但从 VQ 机制看，完整目标很可能还包含类似 commitment 项，用于让 encoder 输出靠近选中的 codeword。这里不展开未核验公式。

---

#### 4. LLM 输入机制

训练好 MQ-Tokenizer 后：

- 用户 ID 被转成若干 user tokens；
- 物品 ID 被转成若干 item tokens；
- 用户历史交互集合被放进 prompt；
- LLM 接收这些 tokenized IDs 和自然语言 prompt。

这种方式把推荐中的结构化 ID 数据转成 LLM 可处理的 token 序列。

---

#### 5. Generative Retrieval：不是生成物品 token，而是生成推荐向量

传统 generative recommendation 常做：

\[
\text{LLM} \rightarrow \text{generate item token sequence}
\]

问题是：

- 需要 autoregressive decoding；
- 需要 beam search；
- 物品空间很大，推理慢；
- 生成非法 ID 或低效搜索的风险高。

TokenRec 改成：

\[
\text{LLM} \rightarrow \mathbf{z}_i
\]

然后在物品向量库中检索：

\[
\text{Top-}K = \text{NearestNeighbor}(\mathbf{z}_i, \{\mathbf{q}_j\}_{j=1}^{m})
\]

因此推荐生成从“文本生成问题”变成“向量检索问题”。

优点：

- 推理更快；
- 避免 beam search；
- 可直接利用 ANN / vector database；
- 更适合大规模 item retrieval。

---

### 关键图表解读

#### 1. 方法框架图：TokenRec 的两阶段结构

输入图中的整体结构说明：

1. 左侧先维护用户和物品的 representation pool；
2. GNN 从用户-物品交互图中学习 collaborative representations；
3. 新用户/新物品可以通过更新 vector database 进入系统；
4. 右侧 TokenRec 使用：
   - User MQ-Tokenizer；
   - Item MQ-Tokenizer；
   - LLM；
   - Generative Retrieval；
5. LLM 输出 generative representation $\mathbf{z}$；
6. 最后通过向量检索得到 top-$K$ item recommendations。

这张图强调了两个点：

- TokenRec 不是纯文本推荐，而是 **LLM + CF representation + vector retrieval**；
- 新用户/新物品泛化依赖 MQ-Tokenizer 对新 representation 的 tokenization 能力。

---

#### 2. $K$ 和 $L$ 的热力图

输入图片中多张 heatmap 横轴是：

\[
K = 1,2,3,4,5
\]

纵轴是：

\[
L = 128,256,512,1024
\]

其中：

- $K$：每个用户/物品使用多少个离散 token；
- $L$：每个 codebook 的大小。

多张图中黑框都标在：

\[
K = 3,\quad L = 512
\]

说明该配置在多个指标/数据集上表现较优。

直观解释：

- $K$ 太小：token 数不够，表达能力不足；
- $K$ 太大：token 序列变长，可能增加噪声和训练难度；
- $L$ 太小：codeword 不够，量化粗糙；
- $L$ 太大：codebook 学习困难，也可能过拟合或稀疏。

因此 $K=3,L=512$ 是表达能力和稳定性的折中。

---

#### 3. Masking Ratio 曲线

输入图片中 masking ratio 与 HR@20 / NDCG@20 的关系显示：

- masking ratio 从 0 增加到约 $0.2$ 或 $0.3$ 时，性能上升；
- 继续增大到 $0.5$ 之后，性能明显下降；
- $0.8$ 附近通常最差。

结论：

> 适度 mask 是正则化，过度 mask 会破坏协同表示的信息。

这说明 MQ-Tokenizer 的泛化不是来自简单噪声，而是依赖“保留足够语义 + 加入适度扰动”的平衡。

---

### 关键贡献

#### 1. 把 ID tokenization 明确为 LLM 推荐的基础问题

论文不是只做一个推荐模型，而是指出：

> LLM-based RecSys 的关键前置问题是：如何把用户/物品 ID 变成 LLM-compatible tokens。

这是一个比具体推荐任务更底层的问题。

---

#### 2. 用 VQ 把 CF 表示离散化

TokenRec 用 MQ-Tokenizer 将 GNN-CF 的连续表示转成离散 token：

\[
\mathbf{q}_j \rightarrow \{w^1_j,\dots,w^K_j\}
\]

这样同时获得：

- LLM 可处理的离散 token；
- CF 模型中的高阶协同知识；
- 比独立 ID token 更小的 token 空间。

---

#### 3. Masking + $K$-way Encoder 提升泛化

两个设计都服务于 generalization：

- masking：让 tokenizer 对表示缺失和噪声更鲁棒；
- $K$-way encoder/codebook：用多个子空间组合表达实体，提升容量和泛化。

---

#### 4. 用 generative retrieval 替代 autoregressive recommendation

TokenRec 不让 LLM 逐 token 生成物品 ID，而是生成推荐向量并检索：

\[
\mathbf{z}_i \rightarrow \text{Top-}K\ \text{items}
\]

这是工程上更可扩展的设计。

---

### 实验与结论

#### 实验设置

根据摘要和图片，实验覆盖：

- 多个真实推荐数据集；
- 图中可见包括 Beauty、Clothing、ML-1M 等；
- 指标包括：
  - HR@20
  - NDCG@20
- 对比对象包括：
  - traditional recommender systems；
  - LLM-based recommender systems；
  - 不同 tokenization 方案；
  - 不同 $K,L$；
  - 不同 masking ratio。

完整表格未在给定内容中展示，因此不写具体数值提升。

---

#### 主要结论

1. **TokenRec 整体优于传统推荐模型和 LLM 推荐基线**  
   摘要明确说明 TokenRec 在综合实验中超过 competitive benchmarks。

2. **MQ-Tokenizer 有效**  
   通过把协同表示量化为离散 token，LLM 能接收更有推荐意义的 ID 表示。

3. **适度 masking 有帮助**  
   masking ratio 约 $0.2$ 到 $0.3$ 时性能最好；过大显著损害性能。

4. **$K=3,L=512$ 是较优配置**  
   多张热力图显示这一组合表现稳定较好。

5. **Generative Retrieval 提升推理效率**  
   摘要说明该范式避免了耗时的 autoregressive decoding 和 beam search。

---

### 局限性

1. **依赖高质量 CF / GNN 表示**  
   如果前置 GNN-CF 表示质量差，MQ-Tokenizer 量化出来的 token 也会受影响。

2. **对新用户/新物品仍需要表示生成机制**  
   图中显示需要更新 vector database，甚至可能需要重新训练 GNN。严格冷启动下，如果没有交互或内容信息，仍然困难。

3. **离散量化会带来信息损失**  
   VQ 把连续表示压缩成有限 codeword，表达能力取决于 $K,L,d_c$ 等超参数。

4. **超参数敏感**  
   热力图说明 $K,L$ 对性能影响明显，masking ratio 也需要调节。

5. **LLM 的实际贡献需要进一步拆解**  
   从方法看，最终推荐依赖向量检索。LLM 生成的 $\mathbf{z}$ 到底比传统 user encoder 强多少，需要依赖完整消融结果判断；给定摘录中未完整展示。

6. **不是纯生成式推荐**  
   TokenRec 名义上是 generative recommendation，但最终输出通过 retrieval 得到，本质是生成向量 + 检索排序的混合范式。

---

### 放进大模型基础知识体系里怎么理解

#### 1. 这是“非文本实体 tokenization”的问题

LLM 原生处理自然语言 token，但推荐系统处理的是：

- user ID；
- item ID；
- interaction graph；
- collaborative signal。

TokenRec 属于一类重要方向：

> 如何把结构化、非语言实体映射到 LLM token space？

类似问题也存在于：

- molecule tokenization；
- graph tokenization；
- code symbol tokenization；
- robot action tokenization；
- multimodal patch/token 表示。

---

#### 2. 这是 VQ-VAE 思想在推荐 ID 上的应用

MQ-Tokenizer 的本质是：

\[
\text{continuous CF embedding}
\rightarrow
\text{discrete codebook indices}
\rightarrow
\text{LLM-compatible tokens}
\]

这和 VQ-VAE 的核心思想一致：

- encoder 输出连续 latent；
- 最近邻查 codebook；
- decoder 重构；
- 用 straight-through estimator 训练不可导的离散选择。

区别是 TokenRec 的输入不是图像 patch 或音频 latent，而是推荐系统里的 user/item collaborative representation。

---

#### 3. 这是“生成 + 检索”的推荐范式

TokenRec 没有让 LLM 直接生成最终 item ID，而是：

\[
\text{LLM生成偏好向量}
+
\text{向量数据库检索}
\]

这和 RAG / dense retrieval 的思想接近：

- LLM 负责生成 query-like representation；
- 检索系统负责从大规模候选中找结果。

对于大规模推荐，这比纯自回归生成更现实。

---

#### 4. 它连接了三个系统

TokenRec 可以看成三者结合：

\[
\text{GNN-CF}
+
\text{VQ Tokenizer}
+
\text{LLM Retrieval}
\]

分别对应：

- GNN-CF：负责学习协同结构；
- VQ：负责离散 token 化；
- LLM：负责用 prompt 和 tokenized ID 建模偏好；
- Retrieval：负责高效 top-$K$ 推荐。

---

### 我需要记住什么

1. **LLM 推荐的基础难点之一是 ID tokenization**：用户和物品不是自然语言，不能简单塞进 LLM。

2. **独立 ID token 会导致词表爆炸**，文本标题又缺少 collaborative filtering 信号。

3. **TokenRec 的核心思想**：

\[
\text{GNN-CF 表示}
\rightarrow
\text{Masked VQ}
\rightarrow
\text{离散 ID tokens}
\rightarrow
\text{LLM}
\rightarrow
\text{Generative Retrieval}
\]

4. **MQ-Tokenizer 的三个关键组件**：
   - masking operation；
   - $K$-way encoder/codebook；
   - $K$-to-1 decoder。

5. **每个物品不是一个 token，而是一组 token**：

\[
v_j \rightarrow \{w^1_j,w^2_j,\dots,w^K_j\}
\]

6. **适度 masking 提升泛化，过度 masking 损害信息**。

7. **Generative Retrieval 是效率关键**：LLM 生成向量，不自回归生成 item ID，再用向量库检索 top-$K$。

8. **从大模型基础角度看，TokenRec 是 VQ 离散化 + 非文本实体 tokenization + retrieval-based generation 的结合。**
