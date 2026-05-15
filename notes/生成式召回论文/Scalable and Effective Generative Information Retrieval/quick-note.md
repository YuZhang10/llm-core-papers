## Scalable and Effective Generative Information Retrieval

### 决策卡片
- 年份：2024
- 引用数：57
- 与搜索意图相关性：高 —— 论文提出 RIPOR，直接研究 generative information retrieval 如何在大规模标准检索 benchmark 上有效，是生成式文档召回方向的关键论文。
- 是否值得进入精读候选：非常值得。适合先读，用来建立 DocID、prefix-oriented ranking、relevance-based identifier 这些生成式召回基础概念。

### 摘要原文

Recent research has shown that transformer networks can be used as differentiable search indexes by representing each document as a sequences of document ID tokens. These generative retrieval models cast the retrieval problem to a document ID generation problem for each given query. Despite their elegant design, existing generative retrieval models only perform well on artificially-constructed and small-scale collections. This has led to serious skepticism in the research community on their real-world impact. This paper represents an important milestone in generative retrieval research by showing, for the first time, that generative retrieval models can be trained to perform effectively on large-scale standard retrieval benchmarks. For doing so, we propose RIPOR- an optimization framework for generative retrieval that can be adopted by any encoder-decoder architecture. RIPOR is designed based on two often-overlooked fundamental design considerations in generative retrieval. First, given the sequential decoding nature of document ID generation, assigning accurate relevance scores to documents based on the whole document ID sequence is not sufficient. To address this issue, RIPOR introduces a novel prefix-oriented ranking optimization algorithm. Second, initial document IDs should be constructed based on relevance associations between queries and documents, instead of the syntactic and semantic information in the documents. RIPOR addresses this issue using a relevance-based document ID construction approach that quantizes relevance-based representations learned for documents. Evaluation on MSMARCO and TREC Deep Learning Track reveals that RIPOR surpasses state-of-the-art generative retrieval models by a large margin (e.g., 30.5% MRR improvements on MS MARCO Dev Set), and perform better on par with popular dense retrieval models.

### 摘要中文翻译

近期研究表明，transformer 网络可以作为可微搜索索引：每个文档被表示为一串 document ID token，生成式召回模型则把检索问题转化为给定查询后生成文档 ID 的问题。尽管这种设计很优雅，已有生成式召回模型主要只在人工构造的小规模集合上表现较好，因此学界对其真实影响存在怀疑。本文是生成式召回研究中的一个重要里程碑，首次展示生成式召回模型可以在大规模标准检索基准上有效训练和运行。为此，作者提出 RIPOR，一个可适用于任意 encoder-decoder 架构的生成式召回优化框架。RIPOR 基于两个常被忽略的设计点：第一，文档 ID 生成具有顺序解码特性，仅基于完整文档 ID 序列给文档分配准确相关性分数是不够的，因此 RIPOR 引入 prefix-oriented ranking optimization；第二，初始文档 ID 应基于 query-document 相关性关联来构造，而不是基于文档自身的句法或语义信息。RIPOR 通过量化文档的 relevance-based representations 来构造基于相关性的文档 ID。在 MSMARCO 和 TREC Deep Learning Track 上，RIPOR 大幅超过已有生成式召回模型，例如在 MS MARCO Dev Set 上 MRR 提升 30.5%，并达到或接近流行 dense retrieval 模型的表现。

### 这篇论文大概在解决什么

这篇论文解决的是：生成式召回能不能摆脱“小数据集玩具方法”的质疑，在真实大规模检索 benchmark 上达到可竞争性能。

它把文档检索变成 DocID 序列生成，并指出两个关键瓶颈：第一，生成过程是 prefix-by-prefix 的，所以优化也要关注前缀阶段的排序质量；第二，DocID 不应该只是语义或句法编码，而应该反映查询-文档相关性结构。

RIPOR 因此提出 prefix-oriented ranking optimization 和 relevance-based DocID construction，是理解后续 generative retrieval 工作的基础论文。

### 可能需要精读时重点看什么

- DocID 构造：relevance-based document representations 如何学习、如何量化。
- prefix-oriented ranking optimization：为什么完整序列相关性不足，前缀阶段如何施加排序损失。
- 与 DSI、NCI 等早期生成式检索方法的差异。
- 大规模检索设置：MSMARCO 和 TREC DL 上的训练、候选规模、beam search、延迟。
- 与 dense retrieval 的对比：它在哪些指标接近 dense retrieval，哪些地方仍有差距。
- 与推荐召回论文的连接：推荐里的 item ID / semantic ID 设计可以借鉴 RIPOR 的 relevance-based identifier 思路。

