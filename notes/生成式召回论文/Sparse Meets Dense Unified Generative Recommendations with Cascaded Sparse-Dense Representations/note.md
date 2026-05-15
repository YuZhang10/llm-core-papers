## Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations

### 一句话定位

COBRA 是一个把 **生成式推荐的 sparse semantic ID** 和 **稠密检索的 dense vector** 串联起来的统一生成检索框架：先生成粗粒度 ID，再生成细粒度向量，用 coarse-to-fine 方式缓解纯 ID 生成推荐的信息损失问题。

### 基本信息

- 论文：**Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations**
- arXiv：2503.02453
- 时间：2025-03-04
- 作者：Yang Yuhao 等，Baidu Inc.
- 任务：Sequential Recommendation / Generative Retrieval / Dense Retrieval
- 方法名：**COBRA**，全称 **Cascaded Organized Bi-Represented generAtive retrieval**
- 依据说明：笔记基于摘要、TeX 源码摘录和给定图片整理；完整实验表格和部分公式细节未完全可见，因此实验数值不展开，只总结趋势。

### 摘要中文翻译

生成式推荐模型近年来受到关注，它们直接根据用户交互序列预测 item identifier。  
但现有方法通常把 **量化 quantization** 和 **序列建模 sequence modeling** 分成不同阶段，导致明显信息损失，难以达到 sequential dense retrieval 的建模精度和推荐准确率。如何统一生成式推荐和稠密检索仍是关键挑战。

本文提出 **COBRA** 框架，通过级联方式融合 **sparse semantic IDs** 和 **dense vectors**。模型先生成稀疏 ID，再以该 ID 为条件生成稠密向量。端到端训练使 dense representation 可以动态优化，同时捕捉 item 语义信息和用户-物品交互中的协同信号。

推理时，COBRA 使用 **coarse-to-fine** 策略：先生成 sparse ID，再通过生成模型细化为 dense vector。论文还提出 **BeamFusion**，结合 beam search 和 nearest neighbor score，以提高推理灵活性和推荐多样性。公开数据集、离线测试和真实广告平台线上 A/B 实验均显示 COBRA 有稳定收益。

### 研究问题

现有推荐检索范式各有缺陷：

#### 1. 纯生成式推荐的问题

以 TIGER 为代表的 generative retrieval 会把 item 内容通过 RQ-VAE 编码成 semantic ID，例如：

\[
item \rightarrow (t_1, t_2, t_3)
\]

然后 Transformer 直接生成下一个 item 的 ID token 序列。

问题是：

- semantic ID 是离散压缩结果，天然有 **quantization loss**；
- 只靠 sparse ID 难表达细粒度 item 差异；
- 量化阶段和序列建模阶段分离，后续模型不能充分修正前面量化的信息损失；
- 对相似 item 的区分能力弱于 dense retrieval。

#### 2. 纯 dense retrieval 的问题

Sequential dense retrieval 会学习用户向量和 item 向量，再做最近邻检索：

\[
score(u, i) = \mathbf{h}_u^\top \mathbf{e}_i
\]

优点是精度高、细粒度相似性强。  
缺点是：

- 依赖大规模 item embedding 存储；
- 检索成本高；
- 难直接享受生成式模型的序列生成、推理、多样性控制等能力。

#### 3. 本文要解决的基础问题

> 如何在一个生成式推荐框架中，同时保留 sparse semantic ID 的语义压缩能力和 dense vector 的细粒度表达能力？

COBRA 的答案是：  
不要只生成 ID，也不要只做向量检索，而是让模型 **级联生成 sparse ID 和 dense vector**。

### 核心方法

#### 1. Cascaded Sparse-Dense Representation

每个 item 不再只用 semantic ID 表示，而是用两类表示共同描述：

\[
r_i = \left(s_i, \mathbf{d}_i\right)
\]

其中：

- \(s_i\)：sparse semantic ID，粗粒度、离散、可生成；
- \(\mathbf{d}_i\)：dense representation，细粒度、连续、可对比学习。

用户历史序列可写成：

\[
\left[(s_{i_1}, \mathbf{d}_{i_1}), (s_{i_2}, \mathbf{d}_{i_2}), \dots, (s_{i_t}, \mathbf{d}_{i_t})\right]
\]

模型目标是预测下一个 item 的：

\[
(s_{i_{t+1}}, \mathbf{d}_{i_{t+1}})
\]

#### 2. Sparse ID 生成

论文沿用 TIGER 风格的 **Residual Quantized VAE, RQ-VAE** 生成 semantic ID。

大致流程：

1. 抽取 item 的 title、brand、category 等属性文本；
2. 用编码器得到 item 表示；
3. 经过 residual quantization 得到多级离散 token：

\[
s_i = (c_i^1, c_i^2, \dots, c_i^L)
\]

这些 token 表示 item 的粗粒度语义结构。

#### 3. Dense Vector 生成

COBRA 额外引入 trainable Transformer Encoder 生成 dense representation。

与静态 item embedding 不同，COBRA 的 dense vector 是训练中动态优化的，目标是补充 sparse ID 丢失的信息：

\[
\mathbf{d}_i = f_{\theta}(x_i)
\]

其中 \(x_i\) 是 item 原始内容或属性输入。

#### 4. Transformer Decoder 交替预测

COBRA 使用 Transformer Decoder 处理级联表示，并按顺序预测：

1. 先预测下一个 item 的 sparse ID；
2. 再以生成出的 sparse ID 为条件预测 dense vector。

可抽象为：

\[
p(s_{t+1}, \mathbf{d}_{t+1} \mid r_{\leq t})
=
p(s_{t+1} \mid r_{\leq t})
\cdot
p(\mathbf{d}_{t+1} \mid r_{\leq t}, s_{t+1})
\]

这个设计的关键点是：

- sparse ID 提供 coarse semantic sketch；
- dense vector 在 ID 条件下学习 fine-grained detail；
- 先粗后细降低直接生成 dense vector 的难度。

#### 5. 训练目标

训练损失包含两部分：

##### Sparse loss

对 semantic ID token 做交叉熵：

\[
\mathcal{L}_{sparse}
=
-\sum_{l=1}^{L} \log p(c_{t+1}^{l} \mid r_{\leq t})
\]

##### Dense loss

dense vector 用对比学习或近邻匹配目标学习，使预测向量靠近真实目标 item，远离负样本：

\[
\mathcal{L}_{dense}
=
-\log
\frac{
\exp(\hat{\mathbf{d}}_{t+1}^{\top}\mathbf{d}_{t+1}^{+}/\tau)
}{
\sum_{j}
\exp(\hat{\mathbf{d}}_{t+1}^{\top}\mathbf{d}_{j}/\tau)
}
\]

总损失：

\[
\mathcal{L}
=
\mathcal{L}_{sparse}
+
\lambda \mathcal{L}_{dense}
\]

其中 \(\lambda\) 控制 sparse 与 dense 目标的权重。

> 注：具体公式可能与论文完整版本略有差异，这里根据摘录中的 \(L_{sparse}\)、\(L_{dense}\) 和方法描述做抽象表达。

#### 6. 推理机制：Coarse-to-Fine Generation

推理时不是一次性直接找 item，而是两阶段：

1. **Coarse generation**  
   生成候选 sparse semantic ID：

   \[
   \hat{s}_{t+1} = \arg\max_s p(s \mid r_{\leq t})
   \]

2. **Fine refinement**  
   将生成的 sparse ID 追加回输入，进一步生成 dense vector：

   \[
   \hat{\mathbf{d}}_{t+1}
   =
   g_{\theta}(r_{\leq t}, \hat{s}_{t+1})
   \]

3. 用 dense vector 做 nearest neighbor retrieval，得到最终推荐 item。

这使 COBRA 同时具备：

- 生成式检索的候选生成能力；
- dense retrieval 的细粒度排序能力。

#### 7. BeamFusion

普通 beam search 只看生成概率，可能导致候选单一或错过 dense 相似 item。  
COBRA 提出 **BeamFusion**，把：

- beam search score；
- nearest neighbor score；

融合起来做最终选择。

可抽象为：

\[
Score(i)
=
\alpha \cdot Score_{beam}(s_i)
+
(1-\alpha) \cdot Score_{NN}(\mathbf{d}, \mathbf{d}_i)
\]

其中 \(M\) 可能表示每个 beam 下取的近邻数量，\(\tau\) 控制分数融合或采样温度。

核心作用：

- 生成 ID 保证语义相关；
- 近邻分数保证向量精度；
- 通过参数控制 recall 与 diversity 的 trade-off。

### 关键图表解读

#### 1. TIGER vs COBRA 架构对比图

图中左侧是 TIGER：

- item 内容先编码成 sparse ID；
- 用户历史被表示为 ID token 序列；
- Transformer 直接生成 next item 的 ID；
- 缺点是只依赖离散 ID，容易丢失细粒度信息。

右侧是 COBRA：

- 每个 item 同时有 **Coarse ID** 和 **Fine Rep.**；
- 输入序列交替包含 \(ID\) 和 \(Dense\)；
- Transformer Decoder 先预测 \(ID_{733}\)，再预测 \(Dense_{733}\)。

这张图表达了本文主线：  
**把推荐从“生成 item ID”升级为“先生成语义类别，再生成细粒度向量”。**

#### 2. COBRA 总体架构图

图中显示：

- Sparse IDs 来自 Residual Quantization；
- Dense vectors 来自 trainable Transformer Encoder；
- 两种表示一起输入 Transformer Decoder；
- Decoder 交替预测 sparse IDs 和 dense vectors；
- 分别计算：

\[
\mathcal{L}_{sparse}, \quad \mathcal{L}_{dense}
\]

关键点：

> sparse ID 不是最终答案，而是 dense vector 生成的条件。

#### 3. Recall-Diversity 趋势图

图中横轴是 \(\tau\)，左轴是 Recall，右轴是 Diversity；不同颜色代表 \(M=20,30,40,50\)。

观察：

- Recall 随 \(\tau\) 增大先上升，在约 \(0.9 \sim 1.0\) 达到峰值，然后下降；
- Diversity 随 \(\tau\) 增大整体下降；
- 更大的 \(M\) 通常带来更高 Recall 和 Diversity；
- 存在明显 accuracy-diversity trade-off。

说明 BeamFusion 不是单纯追求最高相关性，而是提供了可调的推荐多样性机制。

#### 4. Cosine similarity 热力图

给出的几张 heatmap 展示 COBRA 生成的广告 item dense representation 之间的余弦相似度。

现象：

- 对角线为高相似度，符合自身相似性；
- 局部块状结构说明相似 item 被聚集到相近区域；
- 含 dense 表示的 COBRA 图中结构更清晰；
- 差异图显示加入 ID / dense 机制后，相似性结构发生明显调整。

这说明 COBRA 的 dense representation 不只是随机 embedding，而学到了 item 间语义和协同关系。

#### 5. 表示可视化图

可视化图中不同商品 / 内容类型形成多个聚类，例如：

- 游戏类；
- 法律 / 文本类；
- 服装鞋靴类；
- 视频广告类。

这表明 COBRA 的 representation 具有一定语义可分性。  
但注意：可视化只能辅助说明结构，不能单独证明推荐效果。

### 关键贡献

1. **提出 cascaded sparse-dense representation**
   - item 同时用 semantic ID 和 dense vector 表示；
   - 缓解纯 semantic ID 的信息损失。

2. **统一 generative retrieval 和 dense retrieval**
   - 先生成 ID；
   - 再生成 dense vector；
   - 最后用 dense vector 做更细粒度检索。

3. **端到端学习 dense representation**
   - dense vector 不是固定 embedding；
   - 可通过推荐目标动态优化；
   - 同时吸收语义信息和 collaborative signals。

4. **提出 coarse-to-fine 推理**
   - sparse ID 负责粗召回；
   - dense vector 负责细排序 / 精检索。

5. **提出 BeamFusion**
   - 融合 beam search 和 nearest neighbor scores；
   - 可调节准确率与多样性。

### 实验与结论

根据摘要和图示，论文实验包括：

- public datasets；
- offline tests；
- real-world advertising platform online A/B tests；
- 平台日活超过 200 million users。

主要结论：

1. COBRA 在推荐准确率上优于已有生成式推荐方法；
2. dense vector 的加入有效补偿 sparse ID 的信息损失；
3. BeamFusion 可以在 Recall 和 Diversity 之间提供灵活折中；
4. 可视化和相似度矩阵说明 learned representation 具有语义聚类结构；
5. 线上 A/B 测试显示在真实广告系统中也有收益，说明方法不只是 benchmark 有效。

由于完整实验表格未给出，这里不列具体指标数值。

### 局限性

1. **方法复杂度高于纯 ID 生成**
   - 需要 sparse ID 生成模块；
   - 需要 dense encoder；
   - 需要 decoder 交替预测；
   - 推理还涉及 nearest neighbor retrieval。

2. **仍依赖量化质量**
   - 如果 RQ-VAE 生成的 semantic ID 质量差，coarse stage 可能限制后续 dense refinement。

3. **推理链路更长**
   - 先生成 ID，再生成 dense vector，再近邻检索；
   - 延迟和工程复杂度可能高于纯 generative retrieval。

4. **BeamFusion 参数敏感**
   - \(\tau\)、\(M\) 等参数会影响 recall-diversity trade-off；
   - 需要根据业务目标调参。

5. **公开细节有限**
   - 当前摘录未完整展示所有数据集、基线、消融表和线上指标；
   - 对实验强度的判断应以完整论文为准。

### 放进大模型基础知识体系里怎么理解

这篇论文可以放在三个基础主题下理解。

#### 1. Tokenization vs Representation

LLM 的核心是 token 序列建模。  
生成式推荐也试图把 item 变成 token：

\[
item \rightarrow semantic\ ID\ tokens
\]

但 item 不像自然语言 token 那样天然离散。强行离散化会损失信息。

COBRA 的启发是：

> 推荐系统不能只做 tokenization，还要保留 continuous representation。

这对应大模型里的一个重要问题：  
**离散 token 适合生成，连续向量适合相似性建模。**

#### 2. Generative Retrieval vs Dense Retrieval

生成式检索：

\[
query \rightarrow docid/itemid
\]

dense retrieval：

\[
query \rightarrow vector \rightarrow ANN
\]

COBRA 是两者的混合：

\[
query
\rightarrow sparse\ ID
\rightarrow dense\ vector
\rightarrow item
\]

它不是简单拼接两个系统，而是让 dense vector 成为生成过程的一部分。

#### 3. Coarse-to-Fine Modeling

COBRA 的基本思想类似多阶段推理：

1. 先确定大方向；
2. 再补充细节；
3. 最后精确匹配。

这在大模型中也很常见，例如：

- hierarchical decoding；
- chain-of-thought 的分步推理；
- coarse plan then fine generation；
- retrieval-augmented generation 中先召回后精排。

### 我需要记住什么

- 纯生成式推荐的问题：**semantic ID 离散压缩导致信息损失**。
- 纯 dense retrieval 的问题：**精度高但存储和检索成本高，生成能力弱**。
- COBRA 的核心：  
  \[
  item = sparse\ ID + dense\ vector
  \]
- 推理顺序：  
  \[
  \text{生成 sparse ID}
  \rightarrow
  \text{生成 dense vector}
  \rightarrow
  \text{nearest neighbor retrieval}
  \]
- Sparse ID 提供 coarse semantic condition；dense vector 提供 fine-grained discrimination。
- BeamFusion 用来融合生成分数和近邻分数，控制 recall 与 diversity。
- 这篇论文的基础意义：  
  **它试图打通推荐系统中的离散生成范式和连续向量检索范式。**
