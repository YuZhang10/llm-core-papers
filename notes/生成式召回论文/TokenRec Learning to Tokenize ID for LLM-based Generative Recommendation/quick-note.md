## TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation

### 决策卡片
- 年份：2024
- 引用数：51
- 与搜索意图相关性：高 —— 论文聚焦 LLM-based generative recommendation 中用户和 item 如何离散 token 化，是生成式召回能否稳定落地的核心组件问题。
- 是否值得进入精读候选：值得。适合重点看“ID tokenization”“协同过滤表示如何离散化为 LLM 兼容 token”“如何避免自回归解码和 beam search 的高延迟”。

### 摘要原文

There is a growing interest in utilizing large-scale language models (LLMs) to advance next-generation Recommender Systems (RecSys), driven by their outstanding language understanding and in-context learning capabilities. In this scenario, tokenizing (i.e., indexing) users and items becomes essential for ensuring a seamless alignment of LLMs with recommendations. While several studies have made progress in representing users and items through textual contents or latent representations, challenges remain in efficiently capturing high-order collaborative knowledge into discrete tokens that are compatible with LLMs. Additionally, the majority of existing tokenization approaches often face difficulties in generalizing effectively to new/unseen users or items that were not in the training corpus. To address these challenges, we propose a novel framework called TokenRec, which introduces not only an effective ID tokenization strategy but also an efficient retrieval paradigm for LLM-based recommendations. Specifically, our tokenization strategy, Masked Vector-Quantized (MQ) Tokenizer, involves quantizing the masked user/item representations learned from collaborative filtering into discrete tokens, thus achieving a smooth incorporation of high-order collaborative knowledge and a generalizable tokenization of users and items for LLM-based RecSys. Meanwhile, our generative retrieval paradigm is designed to efficiently recommend top-$K$ items for users to eliminate the need for the time-consuming auto-regressive decoding and beam search processes used by LLMs, thus significantly reducing inference time. Comprehensive experiments validate the effectiveness of the proposed methods, demonstrating that TokenRec outperforms competitive benchmarks, including both traditional recommender systems and emerging LLM-based recommender systems.

### 摘要中文翻译

由于大规模语言模型具有出色的语言理解和上下文学习能力，越来越多研究尝试用 LLM 推进下一代推荐系统。在这一场景中，对用户和 item 进行 token 化或索引化，是让 LLM 与推荐任务顺利对齐的关键。已有研究尝试通过文本内容或隐式表示来表示用户和 item，但如何把高阶协同知识高效捕获为与 LLM 兼容的离散 token 仍是挑战。此外，大多数现有 tokenization 方法难以有效泛化到训练语料中未出现的新用户或新 item。为解决这些问题，作者提出 TokenRec，它不仅包含有效的 ID tokenization 策略，也包含面向 LLM 推荐的高效检索范式。具体来说，TokenRec 提出 Masked Vector-Quantized Tokenizer，将协同过滤学到的 masked 用户/item 表示量化为离散 token，从而把高阶协同知识平滑融入 LLM 推荐，并获得可泛化的用户和 item tokenization。同时，它的生成式召回范式被设计为能够高效推荐 top-K item，避免 LLM 中耗时的自回归解码和 beam search，大幅降低推理时间。实验表明，TokenRec 优于传统推荐系统和新兴 LLM-based 推荐系统等竞争基线。

### 这篇论文大概在解决什么

这篇论文解决的是：LLM 做生成式推荐时，用户和 item 不能直接当普通文本处理，必须有能表达协同过滤知识、又能被 LLM 使用的离散 token。

TokenRec 的核心是 MQ Tokenizer：从协同过滤表示中学习可量化的用户/item token，让 token 不只是语义标签，还携带高阶协同信号。同时，论文尝试绕开传统 LLM 自回归逐 token 解码和 beam search 的高推理成本，直接面向 top-K 推荐设计检索范式。

### 可能需要精读时重点看什么

- MQ Tokenizer 如何从 masked user/item representations 中学习离散 token。
- token 是否支持 unseen user/item 的泛化，以及泛化机制是什么。
- top-K 生成式召回如何避免自回归解码和 beam search。
- 与 semantic ID 方法的区别：TokenRec 的 token 更偏协同过滤知识，而不只是内容语义。
- 与 EAGER 的关系：EAGER 关心 behavior/semantic 双流协同，TokenRec 更关心 tokenization 本身。
- 如果你要做生成式召回系统，TokenRec 适合作为“ID 体系设计”的重点参考。

