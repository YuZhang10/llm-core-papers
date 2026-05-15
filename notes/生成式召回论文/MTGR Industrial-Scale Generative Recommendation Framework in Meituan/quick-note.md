## MTGR: Industrial-Scale Generative Recommendation Framework in Meituan

### 决策卡片
- 年份：2025
- 引用数：81
- 与搜索意图相关性：高 —— 论文是美团主流量场景的工业级生成式推荐框架，重点讨论如何把生成式推荐扩展到真实大规模召回/推荐链路。
- 是否值得进入精读候选：非常值得。尤其适合看“工业系统如何保留传统推荐特征”“HSTU 架构如何用于生成式推荐”“训练与推理如何加速”。

### 摘要原文

Scaling law has been extensively validated in many domains such as natural language processing and computer vision. In the recommendation system, recent work has adopted generative recommendations to achieve scalability, but their generative approaches require abandoning the carefully constructed cross features of traditional recommendation models. We found that this approach significantly degrades model performance, and scaling up cannot compensate for it at all. In this paper, we propose MTGR (Meituan Generative Recommendation) to address this issue. MTGR is modeling based on the HSTU architecture and can retain the original deep learning recommendation model (DLRM) features, including cross features. Additionally, MTGR achieves training and inference acceleration through user-level compression to ensure efficient scaling. We also propose Group-Layer Normalization (GLN) to enhance the performance of encoding within different semantic spaces and the dynamic masking strategy to avoid information leakage. We further optimize the training frameworks, enabling support for our models with 10 to 100 times computational complexity compared to the DLRM, without significant cost increases. MTGR achieved 65x FLOPs for single-sample forward inference compared to the DLRM model, resulting in the largest gain in nearly two years both offline and online. This breakthrough was successfully deployed on Meituan, the world's largest food delivery platform, where it has been handling the main traffic.

### 摘要中文翻译

Scaling law 已在自然语言处理、计算机视觉等领域得到广泛验证。在推荐系统中，近期工作尝试用生成式推荐实现可扩展性，但这类方法通常需要舍弃传统推荐模型中精心构造的交叉特征。作者发现，这会显著损害模型性能，单纯扩大模型也无法弥补。为解决这一问题，本文提出 MTGR（Meituan Generative Recommendation）。MTGR 基于 HSTU 架构建模，同时能够保留原始 DLRM 特征，包括交叉特征。此外，MTGR 通过用户级压缩实现训练与推理加速，以确保可扩展性。作者还提出 Group-Layer Normalization，用于增强不同语义空间中的编码效果，并提出 dynamic masking 避免信息泄漏。作者进一步优化训练框架，使模型在计算复杂度达到 DLRM 的 10 到 100 倍时，也不会显著增加成本。MTGR 单样本前向推理达到 DLRM 的 65 倍 FLOPs，并取得近两年来最大的离线和线上收益。该突破已部署在美团这个全球最大外卖平台的主流量中。

### 这篇论文大概在解决什么

这篇论文主要解决的是：工业推荐系统里，生成式推荐如何在保留传统 DLRM 强特征能力的同时实现规模化。

很多生成式推荐工作会把用户和 item 简化为序列 token，获得可扩展建模能力，但代价是丢掉传统推荐里长期积累的 cross features。MTGR 的核心立场是：生成式推荐不能靠“扔掉特征然后扩大模型”解决一切，必须兼容工业特征系统。

因此它把 HSTU 生成式序列建模、DLRM 特征、用户级压缩、GLN、dynamic masking 和训练系统优化组合起来，形成可上线的工业框架。

### 可能需要精读时重点看什么

- HSTU 如何接入推荐特征，和普通 transformer/decoder 架构有什么差别。
- DLRM cross features 如何保留到生成式模型中，输入表示是否可迁移。
- user-level compression 如何降低训练和推理成本。
- GLN 的动机：不同语义空间编码为什么需要 group-level normalization。
- dynamic masking 如何避免训练中的信息泄漏。
- 线上部署细节：主流量服务、延迟、吞吐、收益指标和工程约束。

