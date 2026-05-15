---
name: llm-paper-read
description: 对用户选中的大模型论文调用 GPT-5.5 生成 quick summary 或中文精读笔记。
---

# llm-paper-read

当用户已经选中某篇论文，并要求 quick summary、摘要翻译、相关性判断、精读、翻译、解读、做笔记时使用。

## 工作流

### 阶段 1：Quick Summary

用户还在判断是否要精读时，先运行：

```bash
llm-core-papers/.venv/bin/python llm-core-papers/skills/llm-paper-read/scripts/quick_summary.py \
  --paper "<论文ID或编号>" \
  --query "<搜索意图>" \
  --config llm-core-papers/config.json
```

如果一个主题下需要沉淀候选：

```bash
llm-core-papers/.venv/bin/python llm-core-papers/skills/llm-paper-read/scripts/quick_summary.py \
  --paper "<论文ID或编号>" \
  --query "<搜索意图>" \
  --config llm-core-papers/config.json \
  --output-dir "notes/<主题目录>"
```

如果用户要求把最近一次搜索结果全部扫一遍：

```bash
llm-core-papers/.venv/bin/python llm-core-papers/skills/llm-paper-read/scripts/quick_summary.py \
  --all-search-results \
  --config llm-core-papers/config.json \
  --output-dir "notes/<主题目录>"
```

Quick summary 只会：
   - 获取 arXiv 标题、作者、摘要、年份。
   - 获取 Semantic Scholar 引用数，如果可用。
   - 调用 GPT-5.5 翻译摘要。
   - 调用 GPT-5.5 判断与搜索意图的相关性。
   - 写入 `quick-note.md`，不下载 PDF/source/images。

### 阶段 2：精读

1. 获取论文标识：
   - arXiv ID，例如 `2005.14165`
   - arXiv URL，例如 `https://arxiv.org/abs/2005.14165`
   - 搜索结果编号，例如 `1`

2. 运行脚本：

```bash
llm-core-papers/.venv/bin/python llm-core-papers/skills/llm-paper-read/scripts/read_paper.py \
  --paper "<论文ID或编号>" \
  --config llm-core-papers/config.json
```

如果一个主题下需要每篇论文单独一个 note：

```bash
llm-core-papers/.venv/bin/python llm-core-papers/skills/llm-paper-read/scripts/read_paper.py \
  --paper "<论文ID或编号>" \
  --config llm-core-papers/config.json \
  --output-dir "notes/<主题目录>"
```

3. 脚本会：
   - 获取 arXiv 元数据。
   - 保存 PDF 到本地论文文件夹。
   - 下载 arXiv source，优先读取 TeX 正文。
   - 提取 source 里的图片，必要时把 PDF 图片渲染为 PNG。
   - 调用 GPT-5.5，文本和关键图片一起输入。
   - 默认追加到 `notes/核心论文笔记.md`；如果传入 `--output-dir`，则写到 `notes/<主题目录>/<论文标题>/note.md`。

## 规则

- 使用 `llm-core-papers/.venv/bin/python`，不要直接用系统 `python3` 跑项目脚本。
- 用户还在决策是否精读时，优先用 `quick_summary.py`。
- 只有用户明确要精读时，才用 `read_paper.py` 下载 PDF/source/images 并生成完整 `note.md`。
- 用户要求主题化沉淀时，每篇论文一个文件夹，保留 PDF、source、images 和 note。
- 中文笔记要服务“打基础”，少写空泛评价，多解释为什么这篇论文重要。
- 如果 source/PDF 正文提取失败，则基于标题、作者、摘要、类别生成第一版笔记，并在笔记中注明。
