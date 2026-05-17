# LLM Core Papers

一起读懂 LLM 发展历程中的关键论文。

这个仓库想做的不是“论文列表”，也不是自动摘要合集，而是一份可以长期生长的开放阅读笔记：把 LLM、Agent、Memory、Diffusion、MoE、Retrieval 等方向里真正改变研究脉络的论文，整理成能读、能查、能串起来的学习材料。

如果你也在补 LLM 基础、追踪研究主线，或者想把自己的论文阅读沉淀公开出来，欢迎一起参与。

## 这个仓库在做什么

很多论文单独看都能懂一点，但真正难的是：

- 一篇论文到底在解决什么核心矛盾？
- 它和前后的工作是什么关系？
- 哪些设计是关键，哪些只是工程细节？
- 这条研究线后来为什么继续走下去，或者为什么被替代？
- 读完以后，我们应该把它放进怎样的知识地图里？

`llm-core-papers` 试图围绕这些问题，沉淀三类内容：

- **论文精读笔记**：每篇论文一个目录，保留 `note.md`，尽量用自己的语言讲清楚动机、方法、实验、局限和后续影响。
- **研究脉络地图**：按主题串联论文之间的关系，而不是只堆时间线。
- **图文并茂的学习材料**：保留关键架构图、流程图、实验图，并解释“这张图应该怎么看”。

## 当前阅读主线

```text
notes/
  Diffusion/
    DDPM -> ViT -> DiT

  LLM-Agent-Memory-Layer/
    ReAct / Reflexion / ToT / Voyager / Generative Agents / AutoGen / MetaGPT ...

  生成式召回论文/
    Generative Retrieval / Generative Recommendation / Semantic ID / TokenRec ...

  Multi-Prototype-Retrieval/
    Prototype-based retrieval / CLIP adaptation / Few-shot recognition ...

  无序集合表征与Set-Transformer/
    Deep Sets / Set Transformer / PointNet / permutation-invariant modeling ...

  MoE/
    Mixture-of-Experts and adaptive computation
```

这些方向会继续扩展。我的目标不是一次性覆盖所有论文，而是把每条线里“值得反复回看”的关键节点先读扎实。

## 笔记长什么样

每篇论文的目录通常类似这样：

```text
Paper_Title/
  note.md
  images/
```

`note.md` 会尽量遵循这套风格：

- 先讲这篇论文在研究史上的位置。
- 再回答它真正想解决的问题。
- 用直觉解释方法，而不是堆公式。
- 关键公式只保留必要部分，并解释每个符号背后的含义。
- 图表不是装饰品，每张图都要说明“该看什么”。
- 分清作者结论、实验事实和我自己的理解。
- 记录局限性、适用边界，以及它给后续工作的启发。

我希望这些笔记读起来像一个认真读过论文的人在旁边讲给你听，而不是模型生成的一页摘要。

## 推荐阅读方式

如果你刚开始，可以按下面的顺序读：

1. 先选一个主题目录，比如 `Diffusion/` 或 `LLM-Agent-Memory-Layer/`。
2. 从最基础或最早的论文开始看，比如 DDPM、ViT、ReAct。
3. 先读 `note.md` 的主线解释，再回到原论文看细节。
4. 遇到看不懂的概念，优先看同目录里前置论文的笔记。
5. 最后再看研究脉络类笔记，把单篇论文放回整条线上。

## 欢迎贡献

这个仓库非常欢迎一起建设。你可以贡献：

- 新论文笔记。
- 对已有笔记的纠错、补充和重写。
- 更清晰的研究脉络图。
- 论文之间关系的补充，比如“这篇其实继承了哪篇工作”。
- 更适合初学者的解释、例子或图示。

贡献笔记时，建议遵循：

- 不直接复制论文原文的大段内容。
- 尽量用自己的话复述。
- 公式服务于理解，不为了完整而完整。
- 图片请注明来源，优先使用论文中的关键图。
- 如果有个人判断，请明确写成“我的理解”或“一个可能的视角”。

更详细的协作方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。如果你想新增论文，也可以直接从 [NOTE_TEMPLATE.md](NOTE_TEMPLATE.md) 开始。

## 本地工具

`skills/` 里保留了一些本地辅助脚本，用来搜索论文、下载 PDF/source、抽取图片或生成草稿。它们只是辅助工具，不是这个仓库的核心。

如果要复现脚本环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.example.json config.json
```

配置 API key 时请使用环境变量或本地 `.env`，不要提交任何真实密钥。

```bash
export OPENAI_API_KEY="..."
```

## 关于原始论文文件

为了控制仓库体积，公开仓库默认不提交大体积原始文件：

- `notes/**/*.pdf`
- `notes/**/*.tar.gz`
- `notes/**/source/`
- `notes/**/**/*_source/`

仓库重点保留 Markdown 笔记和精选图片资产。严肃引用、复现实验或核对细节时，请回到论文原文、arXiv、OpenReview、ACL Anthology、会议论文集或项目主页。

## License

笔记内容默认以 [CC BY 4.0](LICENSE) 开放。你可以分享、改写和引用，但请保留来源说明。

论文原文、图片和第三方资料版权归原作者或出版方所有。
