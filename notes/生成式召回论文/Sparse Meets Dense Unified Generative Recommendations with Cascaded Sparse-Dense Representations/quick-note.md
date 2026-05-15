## Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations

### 决策卡片
- 年份：2025
- 引用数：55
- 与搜索意图相关性：高 —— 论文提出 COBRA，把 sparse semantic ID 与 dense vector 级联到一个生成式召回框架里，正面处理生成式召回与 dense retrieval 的融合问题。
- 是否值得进入精读候选：值得。特别适合看“生成 semantic ID 后如何接 dense vector refinement”“beam search 与 nearest neighbor score 如何融合”。

### 摘要原文

Generative models have recently gained attention in recommendation systems by directly predicting item identifiers from user interaction sequences. However, existing methods suffer from significant information loss due to the separation of stages such as quantization and sequence modeling, hindering their ability to achieve the modeling precision and accuracy of sequential dense retrieval techniques. Integrating generative and dense retrieval methods remains a critical challenge. To address this, we introduce the Cascaded Organized Bi-Represented generAtive retrieval (COBRA) framework, which innovatively integrates sparse semantic IDs and dense vectors through a cascading process. Our method alternates between generating these representations by first generating sparse IDs, which serve as conditions to aid in the generation of dense vectors. End-to-end training enables dynamic refinement of dense representations, capturing both semantic insights and collaborative signals from user-item interactions. During inference, COBRA employs a coarse-to-fine strategy, starting with sparse ID generation and refining them into dense vectors via the generative model. We further propose BeamFusion, an innovative approach combining beam search with nearest neighbor scores to enhance inference flexibility and recommendation diversity. Extensive experiments on public datasets and offline tests validate our method's robustness. Online A/B tests on a real-world advertising platform with over 200 million daily users demonstrate substantial improvements in key metrics, highlighting COBRA's practical advantages.

### 摘要中文翻译

近年来，生成式模型因能够从用户交互序列中直接预测 item 标识符而受到推荐系统领域关注。然而，已有方法通常把量化和序列建模等阶段分离，导致显著信息损失，限制了它们达到序列 dense retrieval 技术的建模精度和准确性。如何融合生成式方法与 dense retrieval 仍是关键挑战。为此，作者提出 COBRA 框架，通过级联过程创新性地整合 sparse semantic ID 与 dense vector。该方法先生成 sparse ID，并将其作为条件辅助 dense vector 的生成，在这些表示之间交替生成。端到端训练使 dense 表示可以动态细化，同时捕获 item 语义信息与用户-item 交互中的协同信号。在推理阶段，COBRA 使用 coarse-to-fine 策略：先生成 sparse ID，再通过生成式模型细化为 dense vector。作者还提出 BeamFusion，将 beam search 与最近邻分数结合，以提升推理灵活性和推荐多样性。公开数据集、离线测试以及 2 亿 DAU 级广告平台的线上 A/B 测试验证了方法的实用价值。

### 这篇论文大概在解决什么

这篇论文解决的是：生成式召回和 dense retrieval 各有优势，如何放进一个统一框架里。

纯生成 item ID 容易受离散化和序列建模分离影响，丢掉细粒度协同信息；纯 dense retrieval 又不具备生成式方法的语义 token 组织能力。COBRA 用级联思路先生成 sparse semantic ID，再生成 dense vector，把召回过程设计成 coarse-to-fine。

它的价值在于提供了一条 hybrid generative retrieval 路线：不是在生成式召回和向量召回之间二选一，而是让生成模型参与 dense 表示的细化。

### 可能需要精读时重点看什么

- sparse semantic IDs 的构造方式，以及它们如何作为 dense vector 生成条件。
- 级联生成目标：训练时如何交替或联合优化 sparse 和 dense 表示。
- BeamFusion：beam search 分数与 nearest neighbor 分数如何组合。
- coarse-to-fine 推理流程是否能兼容现有 ANN 服务。
- 在线 A/B 部分：广告平台里的收益指标、延迟要求和候选规模。
- 与你的方向的连接：如果你关心“多路召回融合”“生成式粗召回 + 向量精召回”，这篇非常值得拆。

