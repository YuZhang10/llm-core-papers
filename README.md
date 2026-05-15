# llm-core-papers

个人论文精读笔记库，聚焦 LLM、Agent Memory、生成式召回、Diffusion、MoE 等方向。

这个仓库的主角不是自动总结脚本，而是可长期回看的论文笔记：每篇论文尽量沉淀成一份 `note.md`，用自己的语言串起研究问题、方法直觉、关键图表、实验结论和局限性。

## 内容结构

```text
llm-core-papers/
  notes/
    Diffusion/
      DDPM_Denoising_Diffusion_Probabilistic_Models/
        note.md
      DiT_Scalable_Diffusion_Models_with_Transformers/
        note.md
      ViT_An_Image_is_Worth_16x16_Words/
        note.md
    LLM-Agent-Memory-Layer/
      <paper-title>/
        note.md
        images/
    生成式召回论文/
      <paper-title>/
        note.md
    ...
  skills/
    llm-paper-search/
    llm-paper-read/
```

## 笔记风格

- 先回答“这篇论文到底解决什么问题”。
- 再讲“为什么这个方法是自然的”。
- 公式只保留理解机制必须的部分，不堆推导。
- 尽量保留架构图、流程图、关键实验图，并解释图里该看什么。
- 区分作者结论和自己的理解。
- 记录局限性，而不是只写优点。

## 已整理主题

- `Diffusion/`：DDPM、ViT、DiT 等扩散模型与 Transformer backbone 相关笔记。
- `LLM-Agent-Memory-Layer/`：Agent、长期记忆、多智能体框架、规划与反思相关论文。
- `生成式召回论文/`：生成式推荐、生成式检索、语义 ID、LLM 推荐相关论文。
- `无序集合表征与Set-Transformer/`：Set Transformer、PointNet、集合表征相关论文。
- `Multi-Prototype-Retrieval/`：多原型检索、多模态/动作识别相关论文。
- `MoE/`：Mixture-of-Experts 与扩散语言模型相关笔记。

## 本地原始论文文件

本地阅读时，我通常会在论文目录里保留 PDF、arXiv source 和抽取图片，方便回查。

公开仓库默认忽略这些大体积原始文件：

- `notes/**/*.pdf`
- `notes/**/*.tar.gz`
- `notes/**/source/`
- `notes/**/**/*_source/`

仓库重点保留 Markdown 笔记和精选图片资产。论文原文请优先通过笔记中的 arXiv、ACL Anthology、OpenReview、Proceedings 或项目页链接访问。

## 工具脚本

`skills/` 下保留了一些早期本地辅助脚本，用于搜索候选论文、下载 PDF/source、抽取图片或生成初稿。它们不是仓库的核心内容，也不代表最终笔记质量。

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

## 开源说明

这个仓库是个人学习笔记，不保证覆盖所有相关工作，也不保证每篇笔记都代表论文作者原意。欢迎把它当成阅读路线、问题索引和方法直觉参考；严肃引用请回到论文原文。

