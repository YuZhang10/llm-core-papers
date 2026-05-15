## Scalable and Effective Generative Information Retrieval

### 一句话定位

这篇论文提出 **RIPOR**，把 generative retrieval 从“小规模玩具集合”推进到 **MS MARCO 8.8M passages** 级别：核心是让模型不仅会生成完整 DocID，还要在 **beam search 的每个 prefix 阶段都能把相关文档留住**，并且用 **relevance-based quantization** 构造更适合检索的 DocID。

> 依据说明：笔记基于 arXiv 摘要、TeX 正文摘录和部分图表。部分图片是 ACM 模板/示例图，与论文内容无关；方法结构主要依据正文摘录与可见 encoder-decoder 图。

### 基本信息

- **论文**：Scalable and Effective Generative Information Retrieval
- **arXiv**：2311.09134
- **作者**：Hansi Zeng, Chen Luo, Bowen Jin, Sheikh Muhammad Sarwar, Tianxin Wei, Hamed Zamani
- **时间**：2023-11-15
- **任务**：Generative Information Retrieval / Differentiable Search Index
- **核心方法**：RIPOR = **Relevance-based Identifiers for Prefix-Oriented Ranking**
- **主干模型**：T5-base encoder-decoder
- **主要数据集**：
  - MS MARCO Passage：8.8M passages
  - TREC DL 2019 / 2020

### 摘要中文翻译

近期研究表明，Transformer 可以作为可微分搜索索引：每篇文档被表示成一串 document ID tokens，检索被转化为给定 query 后生成文档 ID 的问题。尽管设计优雅，已有 generative retrieval 模型主要只在人工构造的小规模集合上表现较好，因此其现实价值受到质疑。

本文首次表明，generative retrieval 模型可以在大规模标准检索基准上有效训练。为此，作者提出 **RIPOR**，一个可应用于任意 encoder-decoder 架构的优化框架。RIPOR 基于两个被忽视的基础设计问题：

1. 由于 DocID 是顺序生成的，只保证完整 DocID 的相关性分数正确是不够的；相关文档的每个 DocID prefix 都必须在 beam search 中存活。
2. 初始 DocID 不应主要基于文档的句法/语义相似性，而应基于 query-document relevance association。

RIPOR 分别用 **prefix-oriented ranking optimization** 和 **relevance-based DocID construction** 解决这两个问题。实验显示，RIPOR 在 MS MARCO 和 TREC DL 上大幅超过已有 generative retrieval 模型，例如在 MS MARCO Dev 上 MRR 提升约 30.5%，并达到或接近主流 dense retrieval 模型水平。

### 研究问题

Generative retrieval 的基本形式：

- 每篇文档 \(d\) 被映射为唯一 DocID：

\[
c_d = [c_1^d, c_2^d, \dots, c_L^d]
\]

- 给定 query \(q\)，encoder-decoder 自回归生成 DocID。
- 推理时用 **constrained beam search** 生成合法 DocID，再映射回文档。

这篇论文解决的基础问题是：

> 为什么 generative retrieval 在理论上优雅，但在真实大规模检索上效果差？

作者认为主要有两个原因：

#### 1. Full-sequence ranking 不适配 beam search

已有方法通常优化：

\[
S(q, c_{d^+}) > S(q, c_{d^-})
\]

即只要求完整 DocID 的得分相关文档高于不相关文档。

但 beam search 是逐 token 解码：

- 第 1 步保留 top-\(k\) prefix；
- 第 2 步继续扩展；
- 中途被剪掉的 prefix 永远无法恢复。

所以相关文档要被最终生成，必须满足：

\[
S^i_{\text{prefix}}(q, c_{d^+}) > S^i_{\text{prefix}}(q, c_{d^-}), \quad i=1,\dots,L
\]

也就是说：**每个 prefix 都要排得好**。

#### 2. DocID 构造方式不够“检索相关”

很多 generative retrieval 方法用：

- BERT embedding + hierarchical k-means
- 文档 n-grams
- 语义描述 token

来构造 DocID。

但检索中的“相似”不是普通语义相似，而是 **query-document relevance similarity**。例如两篇文档语言相近，不一定对同一批 query 都相关。

因此 DocID 应该编码的是：

- 哪些文档在检索任务中相关性行为相似；
- 并且最好有层级结构，适配 beam search 的 prefix decoding。

### 核心方法

RIPOR 有两条主线：

1. **Relevance-based DocID Construction**
2. **Prefix-Oriented Ranking Optimization**

#### 1. Generative retrieval scoring

可见图展示了典型 encoder-decoder 架构：

```text
query → Encoder → Decoder
                  ↑
           c0, c1, ..., c_{i-1}
                  ↓
                 h_i
```

第 \(i\) 个 DocID token 的 decoder hidden state：

\[
h_i^d = \text{Decoder}(c_{<i}^d; \text{Encoder}(q))
\]

每个位置有独立的 DocID token embedding table：

\[
E_i \in \mathbb{R}^{V \times D}
\]

论文采用 conditional logit scoring，而不是完整 softmax 概率：

\[
S(q, c_d)
=
\sum_{i=1}^{L} E_i[c_i^d]^\top h_i^d
\]

prefix score 对应前 \(i\) 个 token：

\[
S^i_{\text{prefix}}(q,c_d)
=
\sum_{t=1}^{i} E_t[c_t^d]^\top h_t^d
\]

这样更适合 margin-based ranking loss，也更省计算。

#### 2. Prefix-Oriented Ranking Optimization

传统 MarginMSE：

\[
\mathcal{L}(q,d^+,d^-)
=
\left(
S(q,d^+) - S(q,d^-) - T(q,d^+,d^-)
\right)^2
\]

其中 \(T\) 是 teacher model 给出的 golden margin，通常来自 cross-encoder。

RIPOR 把它分解到 prefix 上：

\[
\mathcal{L}_{\text{rank}}^i
=
\left(
S^i_{\text{prefix}}(q,c_{d^+})
-
S^i_{\text{prefix}}(q,c_{d^-})
-
\alpha_i T(q,d^+,d^-)
\right)^2
\]

其中：

- \(i\)：prefix 长度；
- \(\alpha_i\)：prefix 位置权重；
- \(\alpha_L=1\)，保证完整序列 margin 与 teacher margin 对齐；
- \(\alpha_i\) 设计为单调递增的凹函数，使早期 prefix 也获得足够约束。

核心直觉：

> 不能只让完整 DocID 排对；要让相关文档的每个 prefix 在 beam search 中都不被淘汰。

#### 3. Progressive Training

为了适配从左到右生成，RIPOR 使用 progressive training：

训练顺序大致是：

\[
i = 4 \rightarrow 8 \rightarrow 16 \rightarrow 32
\]

先训练短 prefix，再训练长 prefix。

但只训练当前长度会遗忘短 prefix 的能力，所以使用 multi-objective loss：

\[
\sum_{(q,d^+,d^-)\in D}
\left(
\mathcal{L}_{\text{rank}}^i
+
\sum_{k=1}^{i-1} \mathcal{L}_{\text{rank}}^k
\right)
\]

实际实现中只在若干关键长度上训练，以节省计算。

#### 4. Relevance-Based DocID Construction

RIPOR 先把 encoder-decoder 模型临时当作 dense retriever 来训练。

文档表示：

\[
\mathbf{d}
=
\text{Decoder}(s_0; \text{Encoder}(d))
\]

query 也用类似方式得到表示。

训练目标使用 MarginMSE，并用多阶段 negative sampling：

1. 先用 BM25 top-\(K\) negative；
2. 训练 dense encoder；
3. 用 dense retrieval 找新的 hard negatives；
4. 再训练。

得到文档 relevance representation 后，用 **Residual Quantization, RQ** 构造 DocID。

RQ 的目标是用多个 codebook token embedding 近似文档向量：

\[
\mathbf{d}
\approx
\sum_{i=1}^{L} E_i[c_i^d]
\]

于是每篇文档获得一个 token 序列：

\[
c_d = [c_1^d, \dots, c_L^d]
\]

这比 hierarchical k-means 更直接最小化 representation distortion，也天然形成多级残差结构，适合 prefix decoding。

#### 5. 完整训练流程

RIPOR 的 pipeline：

1. **DocID Initialization**
   - 把 T5 encoder-decoder 当 dense encoder；
   - 用 relevance objective 训练；
   - 对文档向量做 RQ；
   - 得到初始 DocID 和 embedding tables。

2. **Seq2seq Pre-training**
   - 使用 doc2query 给每篇文档生成 pseudo queries；
   - 输入 pseudo query，预测对应 DocID；
   - 用 cross-entropy 训练模型熟悉 corpus。

3. **Rank-Oriented Fine-tuning**
   - 初始 fine-tuning：用 dense negatives，训练 full DocID ranking；
   - prefix-oriented fine-tuning：用 beam search 产生 self negatives，训练 prefix loss；
   - self-negative fine-tuning：再用当前模型生成 hard negatives，做最终 full-length 优化。

#### 6. 推理机制

推理时：

1. 输入 query；
2. encoder 编码 query；
3. decoder 自回归生成 DocID token；
4. 使用 **constrained beam search**，只允许生成合法 DocID prefix；
5. 生成 top-\(K\) DocID；
6. 映射回文档并排序。

RIPOR 的优化目标本质上是让相关文档在每一步 beam search 中都能留下来。

### 关键图表解读

#### Encoder-Decoder 结构图

图中展示：

- query 输入 encoder；
- decoder 接收历史 DocID tokens \(c_0,c_1,\dots,c_{i-1}\)；
- 输出当前位置 hidden state \(h_i\)；
- 与当前位置 token embedding table \(E_i\) 做匹配。

这说明 generative retrieval 不是直接输出自然语言答案，而是输出文档标识符序列。

关键点：

> 文档 ID token 的每个位置都有独立 embedding table，因此 DocID 更像一组结构化 code，而不是普通文本 token。

#### RIPOR 总览图

正文说明该图分为两部分：

1. 上半部分：两个核心组件  
   - relevance-based DocID construction；
   - prefix-oriented ranking optimization。

2. 下半部分：完整优化流程  
   - DocID initialization；
   - seq2seq pre-training；
   - rank-oriented fine-tuning。

#### 实验表

主要结果：

| Model | MS MARCO MRR@10 | MS MARCO Recall@10 |
|---|---:|---:|
| DSI | .045 | .138 |
| DSI-QG | .105 | .292 |
| NCI-QG | .153 | .352 |
| MINDER | .186 | .383 |
| LTRGR | .255 | .531 |
| **RIPOR** | **.333** | **.562** |

RIPOR 相比最强 generative baseline LTRGR：

\[
\frac{0.333 - 0.255}{0.255} \approx 30.6\%
\]

即论文摘要中提到的约 30.5% MRR 提升。

与 dense retrieval 对比：

| Model | MS MARCO MRR@10 |
|---|---:|
| DPR | .287 |
| ANCE | .301 |
| MarginMSE | .312 |
| TAS-B | .323 |
| **RIPOR** | **.333** |

RIPOR 在 MS MARCO Dev 上超过这些 dense retrieval baseline；但在 TREC DL 的部分 recall 指标上仍不一定全面领先。

### 关键贡献

1. **指出 generative retrieval 的 prefix survival 问题**
   - full DocID 排得好不等于 beam search 能生成出来；
   - 这是 generative retrieval 与 dense retrieval 的核心差异。

2. **提出 prefix-oriented ranking loss**
   - 把 ranking supervision 分解到每个 prefix；
   - 直接优化 beam search 过程中的中间决策。

3. **提出 relevance-based DocID construction**
   - DocID 不再主要来自通用语义聚类；
   - 而是来自检索相关性表示；
   - 使用 residual quantization 形成层级化 token ID。

4. **证明 generative retrieval 可扩展到大规模标准 benchmark**
   - 在 8.8M MS MARCO passages 上显著超过已有 generative retrieval 模型；
   - 达到或接近 dense retrieval 水平。

### 实验与结论

#### 设置

- Backbone：T5-base
- DocID 长度：\(L=32\)
- DocID vocabulary size：\(V=256\)
- pseudo queries：每篇文档 10 个
- 训练硬件：8 张 40GB A100
- 检索集合：MS MARCO 8.8M passages
- 指标：
  - MS MARCO：MRR@10, Recall@10
  - TREC DL：NDCG@10, Recall@10

#### 主要结论

1. **RIPOR 是当前摘录中最强 generative retrieval 模型**
   - MS MARCO MRR@10 = .333；
   - 大幅超过 LTRGR 的 .255。

2. **RIPOR 能和 dense retrieval 竞争**
   - MS MARCO 上超过 DPR、ANCE、MarginMSE、TAS-B；
   - TREC DL 上整体接近 dense retriever，但不是所有指标都领先。

3. **两类设计都重要**
   - prefix-oriented optimization 解决生成过程中的 beam search 剪枝问题；
   - relevance-based DocID 解决文档标识符本身不适合检索的问题。

4. **生成式索引的可行性被加强**
   - 此前 generative retrieval 常被批评只能做小规模/人工数据；
   - RIPOR 证明它可以在真实大规模 passage retrieval 上有竞争力。

### 局限性

1. **训练成本很高**
   - 需要 doc2query；
   - 需要多阶段训练；
   - 需要 hard negative mining；
   - 使用 8×A100，流程复杂。

2. **依赖 teacher signal**
   - MarginMSE 需要 golden margin，通常来自 cross-encoder 或强 teacher；
   - 不是纯端到端自监督。

3. **DocID 固定后更新困难**
   - 新增文档可能需要重新量化、维护合法 DocID 结构；
   - 动态语料场景仍是挑战。

4. **推理仍依赖 constrained beam search**
   - 需要维护合法 prefix 约束；
   - beam size 与召回/延迟之间有 trade-off。

5. **只验证 passage retrieval**
   - 主要实验是 MS MARCO 和 TREC DL；
   - 对长文档、多模态、动态网页搜索等场景还需验证。

6. **不是所有 dense baseline 指标都被超过**
   - 在 TREC DL 2020 Recall@10 等指标上，TAS-B 等 dense retriever 仍更强。

### 放进大模型基础知识体系里怎么理解

这篇论文可以放在三个基础模块之间理解：

#### 1. Retrieval Paradigms

传统路线：

```text
BM25 sparse retrieval
→ dense retrieval
→ cross-encoder reranking
→ generative retrieval
```

Generative retrieval 的特别之处：

- 不显式用倒排索引或向量索引；
- 把 corpus 信息压入模型参数和 DocID embedding；
- 检索变成“生成文档 ID”。

#### 2. Seq2Seq Generation 与 Search

普通 seq2seq 生成关注最终序列概率。

但在检索里，生成过程本身就是搜索过程：

\[
\text{generation} = \text{retrieval}
\]

所以每一步 prefix 决策都影响最终召回。

RIPOR 的重要性在于：

> 它把 IR ranking objective 和 autoregressive decoding mechanism 对齐了。

#### 3. Representation Learning + Quantization

RIPOR 的 DocID 不是随便编号，而是对 relevance representation 做 quantization：

```text
document relevance vector
→ residual quantization
→ structured DocID tokens
```

这和向量检索里的 PQ/RQ 有联系，只是这里量化结果被用于生成式 DocID。

### 我需要记住什么

- Generative retrieval 的核心形式：

\[
q \rightarrow \text{generate DocID} \rightarrow d
\]

- 最大问题不是模型不会给完整 DocID 打分，而是：

> 相关文档的 prefix 可能在 beam search 早期就被剪掉。

- RIPOR 的两个核心创新：

```text
1. Prefix-Oriented Ranking Optimization
2. Relevance-Based DocID Construction
```

- 关键公式：

\[
S(q,c_d)=\sum_{i=1}^{L}E_i[c_i^d]^\top h_i^d
\]

\[
\mathcal{L}_{\text{rank}}^i
=
\left(
S^i_{\text{prefix}}(q,c_{d^+})
-
S^i_{\text{prefix}}(q,c_{d^-})
-
\alpha_i T(q,d^+,d^-)
\right)^2
\]

- DocID 应该编码 **检索相关性结构**，不是普通语义相似性。

- 这篇论文的基础意义：

> 它证明 generative retrieval 不只是小规模概念验证；只要 DocID 构造和训练目标与 beam search 对齐，就能在百万级/千万级检索集合上接近 dense retrieval。
