# 本地论文精读落地计划

这个 plan 用来把一个论文主题目录从“已有 quick-note 的论文清单”推进到“可长期复习的完整本地材料库”。它适合复用到新的论文专题目录，例如 agent、memory、retrieval、alignment、multimodal 等。

## 目标

- 为目标目录下的每篇论文补齐本地原始材料和中文精读笔记。
- 保持每篇论文目录结构一致，方便后续搜索、复习和横向串联。
- 尽量依赖论文 PDF、arXiv source、本地解析和自己的理解，而不是外部总结。

## 输入

- 目标论文目录，例如：
  `llm-core-papers/notes/LLM-Agent-Memory-Layer`
- 每篇论文已有的 `quick-note.md` 或目录名。
- 可选参考样例目录，例如：
  `llm-core-papers/notes/生成式召回论文/EAGER Two-Stream Generative Recommender with Behavior-Semantic Collaboration`
- 可选 manifest，包含 arXiv id、论文标题、目录名。

## 每篇论文的目标结构

```text
<paper-dir>/
  <paper-title>.pdf
  quick-note.md
  note.md
  source/
  images/
    index.md
```

## 执行流程

1. **盘点论文清单**
   - 扫描目标目录下的论文子目录。
   - 从 `quick-note.md`、目录名或用户给定 manifest 中确认论文标题和 arXiv id。
   - 明确本轮要处理的论文数量，避免漏掉或误处理无关目录。

2. **下载原始材料**
   - 下载 `https://arxiv.org/pdf/<id>` 到论文目录根部，命名为可读的论文标题 PDF。
   - 下载 `https://arxiv.org/e-print/<id>` 到 `source/`。
   - 解包 arXiv source，保留 TeX、bib、图片、附录等原始文件。

3. **整理图片材料**
   - 优先从 `source/` 中复制论文原始 figures 到 `images/`。
   - 跳过明显无关的 logo、icon、样式文件和缓存文件。
   - 如果 source 图片不足，再从 PDF 渲染关键页面或图表为 PNG。
   - 生成 `images/index.md`，记录文件名、相对路径、尺寸、格式和来源。

4. **阅读论文**
   - 优先读 arXiv TeX source，因为它更容易定位章节、公式、图表和引用。
   - 用 PDF 交叉检查图表、排版顺序和附录内容。
   - 不只翻译摘要，要抓住研究问题、方法假设、算法流程、实验设计、边界条件和它在知识体系里的位置。

5. **写完整中文 note**
   - `note.md` 建议使用统一结构：
     - `一句话定位`
     - `基本信息`
     - `摘要中文翻译`
     - `研究问题`
     - `核心方法`
     - `关键图表解读`
     - `关键贡献`
     - `实验与结论`
     - `局限性`
     - `放进大模型基础知识体系里怎么理解`
     - `我需要记住什么`
   - 在 `关键图表解读` 中嵌入本地图片，使用相对路径。
   - 保留 `quick-note.md` 不改，`note.md` 承担精读和复习功能。

6. **补个人理解卡点**
   - 阅读过程中如果出现“第一次读没理解”的问题，要直接落进 `note.md`。
   - 典型内容包括：符号含义、算法停止条件、是否有环境交互、和相邻论文的区别、工程上怎么类比。
   - 这些卡点比标准摘要更有复习价值。

7. **最终校验**
   - 每篇论文目录应该包含且只包含一个主 PDF。
   - `source/` 非空。
   - `images/index.md` 存在。
   - `note.md` 非空，并包含统一章节。
   - 本地图片链接能从 `note.md` 正确指向 `images/`。

## 复用时的注意事项

- 如果用户明确要求“不调用 skill”，就只用 shell、下载、解包、PDF/TeX 读取和自己的理解。
- 不要覆盖用户已有的 `quick-note.md`。
- 如果某篇论文没有 arXiv source，要在 note 中说明材料来源限制，但不要泛泛写“依据有限”。
- 对同一主题下的多篇论文，最后最好再写一个主线串联：按问题演化、方法层级、memory/state 位置或工程范式组织，而不是按发布时间机械罗列。

## LLM Agent Memory Layer 实例

这个 plan 已用于：

`llm-core-papers/notes/LLM-Agent-Memory-Layer`

当时处理的论文包括 ReAct、Zero-Shot Planners、Reflexion、HuggingGPT、Self-Refine、Generative Agents、Tree of Thoughts、Voyager、MetaGPT、AutoGen。主线可以概括为：

```text
LLM 从一次性回答器，逐步变成有状态的 agent：
行动 grounding -> 任务内思考状态 -> 失败经验反思 -> 长期记忆 -> 工具/多 agent 共享状态
```
