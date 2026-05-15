#!/usr/bin/env python3
"""Create a lightweight decision note for one arXiv paper."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from read_paper import (
    fetch_arxiv_metadata,
    gpt55_request,
    load_config,
    normalize_arxiv_id,
    project_root,
    safe_filename,
)


def load_search_query(config: dict[str, Any]) -> str:
    results_path = project_root() / config.get("search_results_file", "notes/search_results.json")
    if not results_path.exists():
        return ""
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(data.get("query") or "")


def load_search_papers(config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    results_path = project_root() / config.get("search_results_file", "notes/search_results.json")
    if not results_path.exists():
        raise SystemExit(f"Search results not found: {results_path}")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return str(data.get("query") or ""), list(data.get("papers") or [])


def normalize_s2_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_count": data.get("citationCount"),
        "influential_citation_count": data.get("influentialCitationCount"),
        "venue": data.get("venue"),
        "year": data.get("year"),
        "semantic_scholar_url": data.get("url"),
    }


def looks_like_arxiv_id(paper_id: str) -> bool:
    return bool(paper_id and __import__("re").match(r"^\d{4}\.\d{4,5}(v\d+)?$", paper_id))


def semantic_scholar_lookup(arxiv_id: str, title: str, api_key: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    fields = "citationCount,influentialCitationCount,venue,year,url"
    if looks_like_arxiv_id(arxiv_id):
        clean_id = arxiv_id.split("v", 1)[0]
        arxiv_url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            + urllib.parse.quote(f"ARXIV:{clean_id}")
            + f"?fields={fields}"
        )
        try:
            req = urllib.request.Request(arxiv_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
            result = normalize_s2_data(data)
            if result.get("citation_count") is not None:
                time.sleep(0.2 if api_key else 1.0)
                return result
        except Exception:
            pass

    search_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        + urllib.parse.urlencode({"query": title, "limit": 1, "fields": fields})
    )
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        time.sleep(0.2 if api_key else 1.0)
        papers = data.get("data") if isinstance(data, dict) else None
        if papers:
            return normalize_s2_data(papers[0])
    except Exception as exc:
        return {
            "citation_count": None,
            "influential_citation_count": None,
            "venue": None,
            "year": None,
            "semantic_scholar_url": None,
            "s2_error": str(exc),
        }
    return {
        "citation_count": None,
        "influential_citation_count": None,
        "venue": None,
        "year": None,
        "semantic_scholar_url": None,
    }


def openalex_lookup(title: str) -> dict[str, Any]:
    url = (
        "https://api.openalex.org/works?"
        + urllib.parse.urlencode(
            {
                "search": title,
                "per-page": 1,
                "select": "id,display_name,cited_by_count,publication_year,primary_location",
            }
        )
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            return {}
        work = results[0]
        source = (
            ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
            if isinstance(work.get("primary_location"), dict)
            else None
        )
        return {
            "openalex_citation_count": work.get("cited_by_count"),
            "openalex_year": work.get("publication_year"),
            "openalex_venue": source,
            "openalex_url": work.get("id"),
        }
    except Exception as exc:
        return {"openalex_error": str(exc)}


def enrich_with_openalex_if_needed(s2: dict[str, Any], title: str) -> dict[str, Any]:
    if s2.get("citation_count") is not None:
        return s2
    openalex = openalex_lookup(title)
    if openalex.get("openalex_citation_count") is None:
        s2.update(openalex)
        return s2
    s2["citation_count"] = openalex.get("openalex_citation_count")
    s2["year"] = s2.get("year") or openalex.get("openalex_year")
    s2["venue"] = s2.get("venue") or openalex.get("openalex_venue")
    s2["semantic_scholar_url"] = s2.get("semantic_scholar_url") or openalex.get("openalex_url")
    s2["citation_source"] = "OpenAlex"
    return s2


def build_prompt(meta: dict[str, Any], s2: dict[str, Any], query: str) -> str:
    authors = ", ".join(meta["authors"])
    year = s2.get("year") or meta["published"][:4]
    citation_count = s2.get("citation_count")
    influential_count = s2.get("influential_citation_count")
    venue = s2.get("venue") or "N/A"
    citation_source = s2.get("citation_source") or ("Semantic Scholar" if citation_count is not None else "N/A")
    paper_id = meta.get("arxiv_id") or meta.get("paper_id") or "N/A"
    source_url = meta.get("abs_url") or meta.get("source_url") or meta.get("url") or "N/A"
    id_label = "arXiv ID" if looks_like_arxiv_id(str(paper_id)) else "Paper ID"
    relevance_instruction = (
        f"搜索意图：{query}\n请判断论文与这个搜索意图的相关性，给出：高/中/低 + 一句话原因。"
        if query
        else "未提供搜索意图。相关性部分请写：未评估，并说明需要 query。"
    )

    return f"""你是一个帮助我筛选是否要精读论文的助手。

请基于标题、作者、摘要和元数据生成一份极简中文 quick note。要求：

1. 不要扩展到论文正文，不要假装读过全文。
2. 必须包含英文摘要原文。
3. 必须给出流畅中文摘要翻译。
4. 必须判断与搜索意图是否相关；如果没有搜索意图，就明确写未评估。
5. 输出 Markdown，从二级标题开始，不要 YAML frontmatter。
6. 目的只是辅助我决定要不要后续精读。

{relevance_instruction}

论文元信息：
- {id_label}: {paper_id}
- Title: {meta["title"]}
- Authors: {authors}
- Published: {meta["published"]}
- Year: {year}
- Venue: {venue}
- Citation count: {citation_count}
- Citation source: {citation_source}
- Influential citation count: {influential_count}
- Source URL: {source_url}
- Citation metadata URL: {s2.get("semantic_scholar_url") or "N/A"}

Abstract:
{meta["summary"]}

请按这个结构输出：

## {meta["title"]}

### 决策卡片
- 年份：
- 引用数：
- 与搜索意图相关性：
- 是否值得进入精读候选：

### 摘要原文

### 摘要中文翻译

### 这篇论文大概在解决什么

### 可能需要精读时重点看什么
"""


def write_quick_note(content: str, output_dir: str | None, meta: dict[str, Any], config: dict[str, Any]) -> Path:
    if output_dir:
        paper_dir = project_root() / output_dir / safe_filename(meta["title"])
        paper_dir.mkdir(parents=True, exist_ok=True)
        note_path = paper_dir / "quick-note.md"
    else:
        note_path = project_root() / config.get("quick_notes_file", "notes/quick_summaries.md")
        note_path.parent.mkdir(parents=True, exist_ok=True)
        if note_path.exists():
            content = "\n\n---\n\n" + content.strip()
        else:
            content = "# 论文 Quick Summaries\n\n" + content.strip()
    note_path.write_text(content.strip() + "\n", encoding="utf-8")
    return note_path


def generate_one(
    paper: str,
    query: str,
    config: dict[str, Any],
    output_dir: str | None,
    no_s2: bool,
    dry_run: bool,
    meta_override: dict[str, Any] | None = None,
) -> Path | None:
    arxiv_id = normalize_arxiv_id(paper, config)
    meta = meta_override or fetch_arxiv_metadata(arxiv_id)
    meta.setdefault("arxiv_id", arxiv_id)
    meta.setdefault("paper_id", meta.get("arxiv_id") or arxiv_id)
    meta.setdefault("authors", [])
    meta.setdefault("summary", "")
    meta.setdefault("published", "")
    if looks_like_arxiv_id(str(meta.get("arxiv_id", ""))):
        meta.setdefault("abs_url", f"https://arxiv.org/abs/{arxiv_id}")

    s2: dict[str, Any] = {}
    if not no_s2:
        s2_key = config.get("semantic_scholar_api_key") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        s2 = semantic_scholar_lookup(arxiv_id, meta["title"], s2_key)
        s2 = enrich_with_openalex_if_needed(s2, meta["title"])

    if dry_run:
        print(f"Title: {meta['title']}")
        print(f"arXiv: {meta['arxiv_id']}")
        print(f"Published: {meta['published']}")
        print(f"Query: {query or 'N/A'}")
        print(f"Citations: {s2.get('citation_count')}")
        return None

    note = gpt55_request(config, build_prompt(meta, s2, query))
    return write_quick_note(note, output_dir, meta, config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", help="arXiv ID, arXiv URL, or search result index.")
    parser.add_argument("--all-search-results", action="store_true", help="Generate quick notes for every paper in search_results.json.")
    parser.add_argument("--query", help="Search intent used for relevance judgment. Defaults to latest search query.")
    parser.add_argument("--config")
    parser.add_argument("--output-dir", help="Write quick-note.md under notes/<topic>/<paper title>/ without downloading assets.")
    parser.add_argument("--no-s2", action="store_true", help="Skip Semantic Scholar citation metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch metadata and print status without calling GPT-5.5.")
    args = parser.parse_args()

    config = load_config(args.config)
    query = args.query if args.query is not None else load_search_query(config)

    if args.all_search_results:
        search_query, papers = load_search_papers(config)
        query = args.query if args.query is not None else search_query
        if not papers:
            raise SystemExit("No papers found in search_results.json")
        for idx, paper in enumerate(papers, 1):
            arxiv_id = paper.get("arxiv_id") or paper.get("paper_id") or paper.get("title")
            title = paper.get("title", arxiv_id)
            print(f"[{idx}/{len(papers)}] {title}")
            note_path = generate_one(arxiv_id, query, config, args.output_dir, args.no_s2, args.dry_run, paper)
            if note_path:
                print(f"  wrote: {note_path}")
        return 0

    if not args.paper:
        raise SystemExit("--paper is required unless --all-search-results is set")

    note_path = generate_one(args.paper, query, config, args.output_dir, args.no_s2, args.dry_run)
    if note_path:
        print(f"Wrote quick note: {note_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
