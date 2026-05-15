# 生成式召回相关高引用论文

检索日期：2026-05-08  
时间窗口：近 2 年，按 2024-05-08 至 2026-05-08 计算。  
筛选条件：与 Generative Retrieval / Generative Recommendation / 生成式召回强相关，且引用数 > 50。引用数主要来自 Semantic Scholar；少数边界论文用 ResearchGate 或论文页面交叉核对。

> 说明：生成式召回在信息检索里通常指模型直接生成 docid / item id / semantic id；在推荐系统里常表现为用自回归或生成式模型直接产出候选 item，替代或融合传统召回阶段。

## 主清单

| 年份 | 论文 | 引用 | 方向 | 发表信息 | 本地 PDF |
|---|---:|---:|---|---|---|
| 2025 | OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment | 193 | 工业推荐，召回排序一体化 | arXiv, 2025-02-26 | [PDF](pdfs/2025-OneRec.pdf) |
| 2025 | MTGR: Industrial-Scale Generative Recommendation Framework in Meituan | 81 | 工业推荐，HSTU 生成式推荐框架 | CIKM 2025, arXiv 2025-05-24 | [PDF](pdfs/2025-MTGR.pdf) |
| 2024 | EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration | 83 | 序列推荐，行为 token 与语义 token 双流生成 | KDD 2024, 2024-06-20 | [PDF](pdfs/2024-EAGER.pdf) |
| 2025 | Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations | 55 | 稀疏 semantic ID + 稠密向量级联生成 | arXiv, 2025-03-04 | [PDF](pdfs/2025-COBRA.pdf) |
| 2024 | TokenRec: Learning to Tokenize ID for LLM-Based Generative Recommendations | 51 | LLM 生成式推荐，item ID tokenization | TKDE / arXiv, 2024-06-15 | [PDF](pdfs/2024-TokenRec.pdf) |
| 2024 | Scalable and Effective Generative Information Retrieval | 57 | 文档检索，RIPOR，大规模 generative IR | WWW 2024, May 13-17 2024 | [PDF](pdfs/2024-RIPOR-Scalable-Effective-GenIR.pdf) |

## 论文要点

### OneRec

- 作者：Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, Guorui Zhou
- 链接：[Semantic Scholar](https://www.semanticscholar.org/paper/b1ee4940c9654c7ff89dac1cfd9176f8453ee45b) / [arXiv](https://arxiv.org/abs/2502.18965)
- 为什么相关：明确从 generative retrieval-based recommendation 出发，把召回和排序统一成端到端生成式推荐模型。
- 关注点：encoder-decoder、稀疏 MoE、session-wise generation、基于 DPO 的 Iterative Preference Alignment。
- 工业结果：在快手主场景线上带来 watch-time 提升。

### MTGR

- 作者：Ruidong Han, Bin Yin, Shangyu Chen, He Jiang, Fei Jiang, Xiang Li, Chi Ma, Mincong Huang, Xiaoguang Li, Chunzhen Jing, Yueming Han, Meng Zhou, Lei Yu, Chuan Liu, Wei Lin
- 链接：[Semantic Scholar](https://www.semanticscholar.org/paper/af68798a6abb93e82ad4f85234d155c64f6c341f) / [arXiv](https://arxiv.org/abs/2505.18654) / [DOI](https://doi.org/10.1145/3746252.3761565)
- 为什么相关：工业级生成式推荐框架，面向美团主流量场景，目标是把生成式推荐扩展到真实大规模召回。
- 关注点：HSTU 架构、保留 DLRM cross features、user-level compression、Group-Layer Normalization、dynamic masking。
- 工业结果：论文称在美团外卖主流量部署，带来近两年最大的离线和线上收益。

### EAGER

- 作者：Yejin Wang, Jiahao Xun, Ming Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, Zhenhua Dong
- 链接：[Semantic Scholar](https://www.semanticscholar.org/paper/aaa5d85787ce317c83d194f1b4d2ac66fda97b65) / [arXiv](https://arxiv.org/abs/2406.14017) / [DOI](https://doi.org/10.1145/3637528.3671775)
- 为什么相关：把候选 item retrieval 建模为自回归序列生成，是推荐召回方向非常直接的一篇。
- 关注点：双流生成架构，分别解码行为 token 与语义 token；用 confidence-based ranking 融合；引入全局对比学习和语义引导迁移任务。
- 适合精读：semantic ID 与 behavior ID 如何协同。

### COBRA

- 作者：Yuhao Yang, Zhi Ji, Zhaopeng Li, Yi Li, Zhonglin Mo, Yue Ding, Kaibo Chen, Zijian Zhang, Jie Li, Shuanglong Li, Lin Liu
- 链接：[Semantic Scholar](https://www.semanticscholar.org/paper/36485756610616bde975b91c259f9f3e1cdc0369) / [arXiv](https://arxiv.org/abs/2503.02453)
- 为什么相关：将 sparse semantic IDs 与 dense vectors 串联到同一个生成式召回框架中，尝试缓解纯 ID 生成的信息损失。
- 关注点：Cascaded Organized Bi-Represented generAtive retrieval，先生成 sparse ID，再生成 dense vector；BeamFusion 结合 beam search 与近邻分数。
- 工业结果：论文报告在 2 亿 DAU 级广告平台线上 A/B 有提升。

### TokenRec

- 作者：Haohao Qu, Wenqi Fan, Zihuai Zhao, Qing Li
- 链接：[Semantic Scholar](https://www.semanticscholar.org/paper/250d0521a774da4db33f7b81e78d8d34592ce6cf) / [arXiv](https://arxiv.org/abs/2406.10450) / [DOI](https://doi.org/10.1109/TKDE.2025.3599265)
- 为什么相关：聚焦 LLM-based generative recommendation 中 item ID 如何 token 化，这是生成式召回能否稳定落地的核心问题之一。
- 关注点：可学习 ID tokenization，减少 item token 与语义/协同信号错配。

### Scalable and Effective Generative Information Retrieval

- 作者：Hansi Zeng, Chen Luo, Bowen Jin, Sheikh Muhammad Sarwar, Tianxin Wei, Hamed Zamani
- 链接：[OpenReview](https://openreview.net/forum?id=7fw6EAxUI7) / [DBLP](https://dblp.org/rec/conf/www/Zeng0JSWZ24) / [DOI](https://doi.org/10.1145/3589334.3645477)
- 为什么相关：提出 RIPOR，证明 generative retrieval 可以在大规模标准检索 benchmark 上有效，而不只是在小集合上可行。
- 关注点：prefix-oriented ranking optimization、relevance-based DocID construction。
- 边界说明：arXiv 预印本是 2023-11-15，但 WWW 2024 会议版在 2024-05-13 至 2024-05-17，落在本次时间窗口内。

## 边界外但值得看

- From Matching to Generation: A Survey on Generative Information Retrieval  
  Semantic Scholar 引用数约 169，arXiv 日期 2024-04-23，严格按 2024-05-08 截止早了约 15 天，所以没有放入主清单。综述价值很高，适合作为入门阅读。链接：[Semantic Scholar](https://www.semanticscholar.org/paper/4fdf88a4b0677360c333d122699547d8485090f4) / [arXiv](https://arxiv.org/abs/2404.14851)

- TIGER: Recommender Systems with Generative Retrieval  
  生成式推荐召回的代表性早期工作，但主要发表在 2023，超出“最近 2 年”窗口。

## 建议阅读顺序

1. 先读 Scalable and Effective Generative Information Retrieval，理解文档检索里的 DocID、prefix ranking、可扩展性问题。
2. 再读 EAGER 和 TokenRec，抓住推荐场景中 semantic ID / behavior token / ID tokenization 的设计空间。
3. 然后读 OneRec、MTGR、COBRA，看工业系统如何把生成式召回与排序、特征、低延迟推理、线上收益放在一起优化。

