---
name: llm-paper-search
description: 极简搜索最近 4 年大模型核心论文候选，不做推荐，等待用户选择是否精读。
---

# llm-paper-search

当用户要搜索大模型论文、找某个方向的核心论文、或查 arXiv / Semantic Scholar 候选时使用。

## 工作流

1. 提取用户查询词。
2. 运行脚本：

```bash
llm-core-papers/.venv/bin/python llm-core-papers/skills/llm-paper-search/scripts/search_papers.py \
  --query "<用户查询>" \
  --top-n 10 \
  --config llm-core-papers/config.json
```

如果没有 `config.json`，使用 `config.example.json` 的默认值。

3. 向用户展示搜索结果，包含：
   - 编号
   - 标题
   - 作者
   - 年份/发布日期
   - arXiv ID
   - 摘要短摘
   - 链接
   - 引用数如果可用

4. 停在这里，询问用户是否要精读某一篇。

## 规则

- 不主动推荐。
- 不自动精读。
- 不输出“最值得读”这类判断。
- 可以按客观字段排序：标题相关性、日期、引用数。
