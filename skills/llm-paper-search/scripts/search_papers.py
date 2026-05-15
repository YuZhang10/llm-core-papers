#!/usr/bin/env python3
"""Search recent LLM papers with arXiv plus optional Semantic Scholar metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(path: str | None) -> dict[str, Any]:
    default_path = project_root() / "config.example.json"
    config_path = Path(path) if path else project_root() / "config.json"
    if not config_path.exists():
        config_path = default_path
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def arxiv_id_from_url(url: str) -> str:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", url)
    if match:
        return match.group(1).removesuffix(".pdf")
    return url.rsplit("/", 1)[-1].removesuffix(".pdf")


def parse_arxiv(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        paper_id = arxiv_id_from_url(entry.findtext("atom:id", "", ARXIV_NS))
        title = clean_text(entry.findtext("atom:title", "", ARXIV_NS))
        summary = clean_text(entry.findtext("atom:summary", "", ARXIV_NS))
        published = clean_text(entry.findtext("atom:published", "", ARXIV_NS))[:10]
        updated = clean_text(entry.findtext("atom:updated", "", ARXIV_NS))[:10]
        authors = [
            clean_text(author.findtext("atom:name", "", ARXIV_NS))
            for author in entry.findall("atom:author", ARXIV_NS)
        ]
        categories = [
            cat.attrib.get("term", "")
            for cat in entry.findall("atom:category", ARXIV_NS)
            if cat.attrib.get("term")
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        papers.append(
            {
                "arxiv_id": paper_id,
                "title": title,
                "authors": authors,
                "published": published,
                "updated": updated,
                "categories": categories,
                "summary": summary,
                "abs_url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_url": pdf_url or f"https://arxiv.org/pdf/{paper_id}",
            }
        )
    return papers


def search_arxiv(query: str, categories: list[str], years_back: int, max_results: int) -> list[dict[str, Any]]:
    start_date = datetime.now(timezone.utc) - timedelta(days=365 * years_back)
    date_query = f"submittedDate:[{start_date:%Y%m%d}0000 TO {datetime.now(timezone.utc):%Y%m%d}2359]"
    cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query)
        if token.lower() not in {"the", "a", "an", "of", "for", "to", "in", "on", "and"}
    ]
    text_query = " AND ".join(f"all:{token}" for token in tokens) if tokens else f'all:"{query}"'
    search_query = f"({cat_query}) AND {date_query} AND {text_query}"
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return parse_arxiv(response.read().decode("utf-8"))


def simple_score(paper: dict[str, Any], query: str) -> float:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9\-]+", query) if len(w) > 1]
    title = paper["title"].lower()
    summary = paper["summary"].lower()
    score = 0.0
    for word in words:
        if word in title:
            score += 3.0
        if word in summary:
            score += 1.0
    try:
        days = (datetime.now(timezone.utc).date() - datetime.fromisoformat(paper["published"]).date()).days
        score += max(0.0, 2.0 - days / 365.0)
    except ValueError:
        pass
    return round(score, 3)


def s2_lookup(papers: list[dict[str, Any]], api_key: str | None) -> None:
    if not papers:
        return
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    for paper in papers:
        arxiv_id = paper["arxiv_id"].split("v", 1)[0]
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            + urllib.parse.quote(f"ARXIV:{arxiv_id}")
            + "?fields=citationCount,influentialCitationCount,venue,year,url"
        )
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
            paper["citation_count"] = data.get("citationCount")
            paper["influential_citation_count"] = data.get("influentialCitationCount")
            paper["venue"] = data.get("venue")
            paper["year"] = data.get("year")
            paper["semantic_scholar_url"] = data.get("url")
            time.sleep(0.2 if api_key else 1.0)
        except Exception:
            paper["citation_count"] = None


def filter_excluded(papers: list[dict[str, Any]], excluded: list[str]) -> list[dict[str, Any]]:
    lowered = [x.lower() for x in excluded]
    kept = []
    for paper in papers:
        haystack = f"{paper['title']} {paper['summary']}".lower()
        if any(word in haystack for word in lowered):
            continue
        kept.append(paper)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--config")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-results", type=int, default=80)
    parser.add_argument("--no-s2", action="store_true", help="Skip Semantic Scholar metadata.")
    args = parser.parse_args()

    config = load_config(args.config)
    papers = search_arxiv(
        args.query,
        config.get("arxiv_categories", ["cs.CL", "cs.AI", "cs.LG"]),
        int(config.get("years_back", 4)),
        args.max_results,
    )
    papers = filter_excluded(papers, config.get("excluded_keywords", []))
    for paper in papers:
        paper["match_score"] = simple_score(paper, args.query)
    papers.sort(key=lambda p: (p["match_score"], p.get("published", "")), reverse=True)
    papers = papers[: args.top_n]

    s2_key = config.get("semantic_scholar_api_key") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not args.no_s2:
        s2_lookup(papers, s2_key)

    output = {
        "query": args.query,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "years_back": config.get("years_back", 4),
        "papers": papers,
    }
    output_path = project_root() / config.get("search_results_file", "notes/search_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    for idx, paper in enumerate(papers, 1):
        authors = ", ".join(paper["authors"][:4])
        if len(paper["authors"]) > 4:
            authors += ", et al."
        print(f"[{idx}] {paper['title']}")
        print(f"    arXiv: {paper['arxiv_id']} | {paper['published']} | citations: {paper.get('citation_count')}")
        print(f"    authors: {authors}")
        print(f"    url: {paper['abs_url']}")
        print(f"    summary: {paper['summary'][:260]}...")
        print()
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
