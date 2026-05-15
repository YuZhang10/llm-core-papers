## EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration

### 决策卡片
- 年份：2024
- 引用数：83
- 与搜索意图相关性：高 —— 论文直接把候选 item retrieval 建模为自回归序列生成，并围绕行为 token 与语义 token 的协同设计生成式推荐框架。
- 是否值得进入精读候选：非常值得。适合重点看“semantic ID + behavior ID 双路召回”“生成式推荐中的 token 设计”和“如何融合协同信号与语义信号”。

### 摘要原文

Generative retrieval has recently emerged as a promising approach to sequential recommendation, framing candidate item retrieval as an autoregressive sequence generation problem. However, existing generative methods typically focus solely on either behavioral or semantic aspects of item information, neglecting their complementary nature and thus resulting in limited effectiveness. To address this limitation, we introduce EAGER, a novel generative recommendation framework that seamlessly integrates both behavioral and semantic information. Specifically, we identify three key challenges in combining these two types of information: a unified generative architecture capable of handling two feature types, ensuring sufficient and independent learning for each type, and fostering subtle interactions that enhance collaborative information utilization. To achieve these goals, we propose (1) a two-stream generation architecture leveraging a shared encoder and two separate decoders to decode behavior tokens and semantic tokens with a confidence-based ranking strategy; (2) a global contrastive task with summary tokens to achieve discriminative decoding for each type of information; and (3) a semantic-guided transfer task designed to implicitly promote cross-interactions through reconstruction and estimation objectives. We validate the effectiveness of EAGER on four public benchmarks, demonstrating its superior performance compared to existing methods.

### 摘要中文翻译

生成式召回最近成为序列推荐中的一种有前景方法，它将候选 item 召回建模为自回归序列生成问题。然而，已有生成式方法通常只关注 item 信息中的行为侧或语义侧，忽视了二者的互补性，因此效果受限。为解决这一问题，作者提出 EAGER，一个能够无缝整合行为信息与语义信息的生成式推荐框架。作者指出，结合这两类信息有三个关键挑战：需要能够处理两种特征类型的统一生成架构；需要保证每类信息都能充分且独立地学习；还需要促进细粒度交互，以增强协同信息利用。为此，作者提出：第一，双流生成架构，用共享 encoder 和两个独立 decoder 分别解码行为 token 与语义 token，并通过 confidence-based ranking 融合；第二，带 summary token 的全局对比任务，使每类信息的解码更具判别性；第三，语义引导迁移任务，通过重构和估计目标隐式促进跨信息交互。作者在四个公开基准上验证了 EAGER 的有效性，结果优于已有方法。

### 这篇论文大概在解决什么

这篇论文解决的是：生成式推荐里的 item 表示应该只依赖行为协同信号，还是只依赖语义信息？EAGER 的答案是两者都要，而且要分开学、再协同融合。

它把推荐召回看成生成行为 token 和语义 token 的过程。行为 token 更贴近协同过滤信号，语义 token 更有可解释性和泛化能力。双流架构允许二者独立建模，再用 confidence-based ranking 进行融合。

这篇更偏“学术方法 + 推荐召回建模”，不是工业系统论文，但对 semantic ID 设计很有参考价值。

### 可能需要精读时重点看什么

- behavior token 与 semantic token 分别如何构造。
- shared encoder + two decoders 的具体结构，以及是否共享词表或位置编码。
- confidence-based ranking 如何融合两个 decoder 的输出。
- summary token 的 global contrastive task 如何提升判别性。
- semantic-guided transfer task 是否能迁移到多原型、多路召回或 hybrid retrieval。
- 与 TokenRec/COBRA 的关系：EAGER 更强调双流信号协同，TokenRec 更强调 tokenization，COBRA 更强调 sparse-dense 级联。

