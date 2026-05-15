## Matching Compound Prototypes for Few-Shot Action Recognition

### 决策卡片
- 年份：2024
- 引用数：未提供
- 与搜索意图相关性：中 —— 论文涉及 Transformer 生成多类型 prototypes 并进行 prototype matching，但任务是 few-shot action recognition，不是面向正样本簇的 multi-prototype retrieval，也未体现 max_k 路由、ANN union retrieval 或 CLIP/MLLM 检索框架。
- 是否值得进入精读候选：低优先级候选；如果你关注“多 prototype 表示 + 不同匹配策略”的设计，可略读方法部分，否则不必优先精读。

### 摘要原文

The task of few-shot action recognition aims to recognize novel action classes using only a small number of labeled training samples. How to better describe the action in each video and how to compare the similarity between videos are two of the most critical factors in this task. Directly describing the video globally or by its individual frames cannot well represent the spatiotemporal dependencies within an action. On the other hand, naively matching the global representations of two videos is also not optimal since action can happen at different locations in a video with different speeds. In this work, we propose a novel approach that describes each video using multiple types of prototypes and then computes the video similarity with a particular matching strategy for each type of prototypes. To better model the spatiotemporal dependency, we describe the video by generating prototypes that model the multi-level spatiotemporal relations via transformers. There are a total of three types of prototypes. The first type of prototypes are trained to describe specific aspects of the action in the video e.g., the start of the action, regardless of its timestamp. These prototypes are directly matched one-to-one between two videos to compare their similarity. The second type of prototypes are the timestamp-centered prototypes that are trained to focus on specific timestamps of the video. To deal with the temporal variation of actions in a video, we apply bipartite matching to allow the matching of prototypes of different timestamps. The third type of prototypes are generated from the timestamp-centered prototypes, which regularize their temporal consistency while serving as an auxiliary summarization of the whole video. Experiments demonstrate that our proposed method achieves state-of-the-art results on multiple benchmarks.

### 摘要中文翻译

少样本动作识别任务旨在仅使用少量带标签训练样本来识别新的动作类别。在这一任务中，如何更好地描述每个视频中的动作，以及如何比较视频之间的相似度，是两个最关键的因素。直接用全局方式描述视频，或仅通过单独帧来描述视频，都无法很好地表示动作中的时空依赖关系。另一方面，简单地匹配两个视频的全局表示也并不理想，因为动作可能出现在视频中的不同位置，并且具有不同速度。

本文提出了一种新方法：使用多种类型的原型来描述每个视频，并针对每类原型采用特定的匹配策略来计算视频相似度。为了更好地建模时空依赖，作者通过 Transformer 生成能够刻画多层级时空关系的原型来描述视频。方法中共有三类原型。第一类原型用于描述视频中动作的特定方面，例如动作的开始阶段，而不考虑其具体时间戳；这类原型在两个视频之间进行一对一直接匹配以比较相似度。第二类原型是以时间戳为中心的原型，被训练为关注视频中的特定时间点；为了处理视频中动作的时间变化，作者采用二分图匹配，使不同时间戳的原型也能够被匹配。第三类原型由时间戳中心原型生成，用于正则化其时间一致性，同时作为整个视频的辅助摘要。实验表明，该方法在多个基准上达到了最先进结果。

### 这篇论文大概在解决什么

这篇论文面向少样本动作识别，核心问题是：只有少量标注视频时，如何更好地表示视频动作并比较两个视频是否属于同一动作类别。

它的主要思路不是用单一全局 embedding 表示一个视频，而是为每个视频构造多种 prototype：

- 描述动作特定阶段或方面的 prototype；
- 关注特定时间戳的 timestamp-centered prototype；
- 用于时间一致性正则和视频整体摘要的辅助 prototype。

然后针对不同 prototype 类型设计不同的匹配方式，例如一对一匹配、二分图匹配等，以处理动作发生时间和速度变化带来的对齐问题。

与搜索意图的联系主要在“多 prototype 表示”和“相似度匹配”层面；但它不是针对手工构造正样本簇学习 K 个潜在模式 prototype，也不是检索系统或 ANN 召回方法。

### 可能需要精读时重点看什么

- 多类型 prototype 是如何生成的，尤其是 Transformer 如何建模多层级时空关系。
- 三类 prototype 的定义、训练目标和各自作用。
- 不同 prototype 对应的匹配策略：一对一匹配、二分图匹配、辅助摘要匹配。
- 视频相似度最终如何由多种 prototype matching 结果组合得到。
- 是否有可迁移到你的检索意图中的设计，例如：
  - 一个样本/集合用多个 prototype 表示；
  - 不同 prototype 负责不同潜在模式；
  - 相似度不是单一 embedding 点积，而是 prototype-level matching。
- 可快速确认它是否涉及 retrieval、cluster-level supervision、max-k routing 或 ANN；从摘要看，这些并不是本文重点。
