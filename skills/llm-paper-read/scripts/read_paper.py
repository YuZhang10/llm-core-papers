#!/usr/bin/env python3
"""Read one arXiv paper with GPT-5.5 and append a Chinese note."""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
NOTE_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif"}
SKIP_IMAGE_NAME_PARTS = ("logo", "icon", "acmguide", "sample-franklin")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | None) -> dict[str, Any]:
    load_dotenv(project_root() / ".env")
    config_path = Path(path) if path else project_root() / "config.json"
    if not config_path.exists():
        config_path = project_root() / "config.example.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["gpt55_base_url"] = os.environ.get("GPT55_BASE_URL", config.get("gpt55_base_url"))
    config["gpt55_model"] = os.environ.get("GPT55_MODEL", config.get("gpt55_model"))
    config["max_tokens"] = os.environ.get("GPT55_MAX_TOKENS", config.get("max_tokens", 6000))
    return config


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_filename(text: str, max_len: int = 140) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", " ", clean_text(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len].rstrip(" .") or "paper"


def normalize_arxiv_id(value: str, config: dict[str, Any]) -> str:
    value = value.strip()
    if value.isdigit():
        results_path = project_root() / config.get("search_results_file", "notes/search_results.json")
        data = json.loads(results_path.read_text(encoding="utf-8"))
        papers = data.get("papers", [])
        idx = int(value) - 1
        if idx < 0 or idx >= len(papers):
            raise SystemExit(f"Search result index out of range: {value}")
        return papers[idx]["arxiv_id"]
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", value)
    if match:
        return match.group(1).removesuffix(".pdf")
    return value.removeprefix("arXiv:").removesuffix(".pdf")


def fetch_arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": arxiv_id})
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-core-papers/0.1 (paper reading assistant)"},
    )
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                root = ET.fromstring(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code != 429 or attempt == 3:
                return fetch_arxiv_abs_metadata(arxiv_id)
            time.sleep(30 * (attempt + 1))
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt == 3:
                return fetch_arxiv_abs_metadata(arxiv_id)
            time.sleep(10 * (attempt + 1))
    else:
        raise last_exc or RuntimeError(f"Failed to fetch arXiv metadata for {arxiv_id}")
    entry = root.find("atom:entry", ARXIV_NS)
    if entry is None:
        raise SystemExit(f"No arXiv paper found for {arxiv_id}")
    authors = [
        clean_text(author.findtext("atom:name", "", ARXIV_NS))
        for author in entry.findall("atom:author", ARXIV_NS)
    ]
    return {
        "arxiv_id": arxiv_id,
        "title": clean_text(entry.findtext("atom:title", "", ARXIV_NS)),
        "authors": authors,
        "summary": clean_text(entry.findtext("atom:summary", "", ARXIV_NS)),
        "published": clean_text(entry.findtext("atom:published", "", ARXIV_NS))[:10],
        "updated": clean_text(entry.findtext("atom:updated", "", ARXIV_NS))[:10],
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def _meta_contents(page: str, name: str) -> list[str]:
    pattern = rf'<meta\s+name="{re.escape(name)}"\s+content="([^"]*)"\s*/?>'
    return [html.unescape(match) for match in re.findall(pattern, page)]


def fetch_arxiv_abs_metadata(arxiv_id: str) -> dict[str, Any]:
    url = f"https://arxiv.org/abs/{arxiv_id}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-core-papers/0.1 (paper reading assistant)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        page = response.read().decode("utf-8", errors="ignore")

    title_values = _meta_contents(page, "citation_title")
    authors = _meta_contents(page, "citation_author")
    abstract_values = _meta_contents(page, "citation_abstract")
    date_values = _meta_contents(page, "citation_date")
    title = title_values[0] if title_values else ""
    published = date_values[0].replace("/", "-") if date_values else ""
    summary = abstract_values[0] if abstract_values else ""
    if not title:
        raise SystemExit(f"No arXiv paper found for {arxiv_id}")
    return {
        "arxiv_id": arxiv_id,
        "title": clean_text(title),
        "authors": [clean_text(author) for author in authors],
        "summary": clean_text(summary),
        "published": clean_text(published)[:10],
        "updated": clean_text(published)[:10],
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def download_file(url: str, path: Path, timeout: int = 90) -> tuple[bool, str]:
    if path.exists() and path.stat().st_size > 0:
        return True, "already exists"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            path.write_bytes(response.read())
        return True, "downloaded"
    except Exception as exc:
        return False, str(exc)


def safe_extract_tar(tar_path: Path, dest: Path) -> bool:
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            safe_members = []
            for member in tar.getmembers():
                member_path = dest / member.name
                if not str(member_path.resolve()).startswith(str(dest.resolve())):
                    continue
                if member.issym() or member.islnk():
                    continue
                safe_members.append(member)
            try:
                tar.extractall(dest, members=safe_members, filter="data")
            except TypeError:
                tar.extractall(dest, members=safe_members)
        return True
    except tarfile.TarError:
        return False


def download_and_extract_source(arxiv_id: str, source_dir: Path) -> str:
    source_dir.mkdir(parents=True, exist_ok=True)
    if any(source_dir.iterdir()):
        return "arXiv source already exists"

    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
    archive_path = source_dir / "source.tar.gz"
    ok, status = download_file(source_url, archive_path)
    if not ok:
        return f"arXiv source download failed: {status}"

    if safe_extract_tar(archive_path, source_dir):
        archive_path.unlink(missing_ok=True)
        return "arXiv source extracted"

    try:
        tex_path = source_dir / "main.tex"
        tex_path.write_bytes(gzip.decompress(archive_path.read_bytes()))
        archive_path.unlink(missing_ok=True)
        return "arXiv source extracted from gzip tex"
    except Exception as exc:
        return f"arXiv source extraction failed: {exc}"


def strip_tex(tex: str) -> str:
    tex = re.sub(r"%.*", "", tex)
    tex = re.sub(r"\\(cite|ref|label|url|href)(\[[^\]]*\])?\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\n\2\n", tex)
    tex = re.sub(r"\\begin\{(figure|table|equation|align|algorithm)[^}]*\}", "\n", tex)
    tex = re.sub(r"\\end\{[^}]+\}", "\n", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", tex)
    tex = tex.replace("{", " ").replace("}", " ")
    return clean_text(tex)


def extract_source_text(source_dir: Path, max_chars: int) -> tuple[str, str]:
    tex_files = sorted(
        source_dir.rglob("*.tex"),
        key=lambda p: (0 if p.name.lower() in {"main.tex", "paper.tex"} else 1, len(str(p))),
    )
    chunks = []
    for tex_file in tex_files[:20]:
        try:
            raw = tex_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        cleaned = strip_tex(raw)
        if len(cleaned) > 200:
            chunks.append(f"\n\n[{tex_file.relative_to(source_dir)}]\n{cleaned}")
        if sum(len(c) for c in chunks) >= max_chars:
            break
    text = clean_text("\n".join(chunks))[:max_chars]
    if text:
        return text, f"arXiv TeX source ({len(tex_files)} tex files)"
    return "", "arXiv TeX source unavailable"


def extract_pdf_text(pdf_path: Path, max_chars: int) -> tuple[str, str]:
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        chunks = []
        for page in reader.pages[:12]:
            chunks.append(page.extract_text() or "")
        text = clean_text("\n".join(chunks))[:max_chars]
        if text:
            return text, "PDF text extracted by pypdf"
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        text = clean_text(result.stdout)[:max_chars]
        if text:
            return text, "PDF text extracted by pdftotext"
    except Exception as exc:
        return "", f"PDF text extraction unavailable: {exc}"

    return "", "PDF text extraction returned no text"


def copy_source_images(source_dir: Path, images_dir: Path) -> list[dict[str, Any]]:
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf", ".eps", ".svg"}
    figures = []
    seen = set()
    images_dir.mkdir(parents=True, exist_ok=True)

    for src in source_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in image_exts:
            continue
        if src.name == "source.tar.gz" or "logo" in src.name.lower() or "icon" in src.name.lower():
            continue
        rel = src.relative_to(source_dir)
        filename = safe_filename("-".join(rel.parts), 180)
        if src.suffix and not filename.lower().endswith(src.suffix.lower()):
            filename += src.suffix.lower()
        if filename in seen:
            filename = f"{len(seen) + 1}-{filename}"
        seen.add(filename)
        dest = images_dir / filename
        try:
            shutil.copy2(src, dest)
        except Exception:
            continue
        figures.append(
            {
                "filename": filename,
                "path": f"images/{filename}",
                "size": dest.stat().st_size,
                "ext": dest.suffix.lower().lstrip("."),
                "source": "arxiv-source",
            }
        )
        if dest.suffix.lower() == ".pdf":
            figures.extend(render_pdf_figure(dest, images_dir))
    return figures


def render_pdf_figure(pdf_path: Path, images_dir: Path) -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore
    except Exception:
        return []

    rendered = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []

    try:
        for page_idx in range(min(len(doc), 3)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=160)
            filename = f"{pdf_path.stem}-page{page_idx + 1}.png"
            out_path = images_dir / filename
            pix.save(str(out_path))
            rendered.append(
                {
                    "filename": filename,
                    "path": f"images/{filename}",
                    "size": out_path.stat().st_size,
                    "ext": "png",
                    "source": "arxiv-source-pdf-render",
                }
            )
    finally:
        doc.close()
    return rendered


def image_content_parts(images_dir: Path, figures: list[dict[str, Any]], max_images: int) -> list[dict[str, Any]]:
    parts = []
    supported = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    for fig in figures:
        if len(parts) >= max_images:
            break
        path = images_dir / fig["filename"]
        if path.suffix.lower() not in supported or not path.exists() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "auto"},
            }
        )
    return parts


def write_image_index(images_dir: Path, figures: list[dict[str, Any]]) -> Path:
    index_path = images_dir / "index.md"
    lines = ["# 图片索引", "", f"总计：{len(figures)} 张图片", ""]
    sources: dict[str, list[dict[str, Any]]] = {}
    for fig in figures:
        sources.setdefault(fig["source"], []).append(fig)
    for source, items in sources.items():
        lines.extend([f"## 来源: {source}", ""])
        for fig in items:
            lines.extend(
                [
                    f"- 文件名：{fig['filename']}",
                    f"- 路径：{fig['path']}",
                    f"- 大小：{fig['size'] / 1024:.1f} KB",
                    f"- 格式：{fig['ext']}",
                    "",
                ]
            )
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def select_note_figures(figures: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Pick local bitmap figures that render directly in Markdown notes."""
    candidates = []
    seen_names = set()
    for fig in figures:
        ext = str(fig.get("ext", "")).lower()
        name = str(fig.get("filename", ""))
        lower_name = name.lower()
        if ext not in NOTE_IMAGE_EXTS:
            continue
        if any(part in lower_name for part in SKIP_IMAGE_NAME_PARTS):
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        candidates.append(fig)

    def score(fig: dict[str, Any]) -> tuple[int, int, int]:
        source = str(fig.get("source", ""))
        name = str(fig.get("filename", "")).lower()
        source_rank = 0 if "render" in source else 1
        name_rank = 0 if re.search(r"(fig|figure|framework|model|architecture|pipeline|result)", name) else 1
        return (source_rank, name_rank, -int(fig.get("size", 0)))

    return sorted(candidates, key=score)[:limit]


def build_note_figure_markdown(figures: list[dict[str, Any]], limit: int = 10, heading_level: int = 4) -> str:
    selected = select_note_figures(figures, limit)
    if not selected:
        return ""

    heading = "#" * heading_level
    lines = [f"{heading} 原始图表", ""]
    for idx, fig in enumerate(selected, 1):
        filename = str(fig["filename"])
        label = Path(filename).stem.replace("-", " ")
        lines.extend(
            [
                f"{heading}# 图 {idx}: {label}",
                f"![图 {idx}: {label}]({fig['path']})",
                "",
            ]
        )
    return "\n".join(lines).strip()


def note_contains_local_image(content: str) -> bool:
    return bool(re.search(r"!\[[^\]]*\]\([^)]*images/[^)]*\)|!\[\[[^\]]*images/[^\]]*\]\]", content))


def ensure_note_contains_images(content: str, figure_markdown: str) -> str:
    if not figure_markdown or note_contains_local_image(content):
        return content

    marker = "### 关键图表解读"
    if marker in content:
        return content.replace(marker, f"{marker}\n\n{figure_markdown}", 1)
    return f"{content.rstrip()}\n\n### 关键图表解读\n\n{figure_markdown}\n"


def prepare_paper_workspace(meta: dict[str, Any], output_dir: str, max_chars: int, no_pdf: bool) -> dict[str, Any]:
    topic_dir = project_root() / output_dir
    paper_name = safe_filename(meta["title"])
    paper_dir = topic_dir / paper_name
    source_dir = paper_dir / "source"
    images_dir = paper_dir / "images"
    paper_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = paper_dir / f"{paper_name}.pdf"
    pdf_status = "skipped by --no-pdf"
    if not no_pdf:
        ok, pdf_status = download_file(meta["pdf_url"], pdf_path)
        if not ok:
            pdf_path = None
            pdf_status = f"PDF download failed: {pdf_status}"

    source_status = download_and_extract_source(meta["arxiv_id"], source_dir)
    source_text, text_status = extract_source_text(source_dir, max_chars)

    if not source_text and pdf_path:
        source_text, text_status = extract_pdf_text(pdf_path, max_chars)
    if not source_text:
        source_text, text_status = "", f"{text_status}; falling back to abstract"

    figures = copy_source_images(source_dir, images_dir)
    image_index = write_image_index(images_dir, figures)

    return {
        "paper_dir": paper_dir,
        "pdf_path": pdf_path,
        "source_dir": source_dir,
        "images_dir": images_dir,
        "image_index": image_index,
        "figures": figures,
        "paper_text": source_text,
        "extraction_status": f"{text_status}; {pdf_status}; {source_status}; images: {len(figures)}",
    }


def gpt55_request(config: dict[str, Any], prompt: str, image_parts: list[dict[str, Any]] | None = None) -> str:
    env_name = config.get("gpt55_api_key_env", "GPT55_API_KEY")
    api_key = os.environ.get(env_name) or config.get("gpt55_api_key")
    if not api_key:
        raise SystemExit(f"Missing GPT-5.5 API key. Set environment variable {env_name}.")

    base_url = config["gpt55_base_url"]
    url = base_url + "?ak=" + urllib.parse.quote(api_key)
    payload = {
        "stream": False,
        "model": config.get("gpt55_model", "gpt-5.5-2026-04-24"),
        "max_tokens": int(config.get("max_tokens", 6000)),
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + (image_parts or []),
            }
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-TT-LOGID": f"llm-paper-read-{datetime.now():%Y%m%d%H%M%S}"},
        method="POST",
    )
    timeout = int(config.get("gpt55_timeout_seconds", 360))
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    text = extract_response_text(data)
    if not text:
        raise SystemExit("GPT-5.5 response did not contain readable text. Raw response saved nowhere by default.")
    return text


def extract_response_text(data: Any) -> str:
    if isinstance(data, dict):
        if isinstance(data.get("choices"), list) and data["choices"]:
            message = data["choices"][0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        if isinstance(data.get("data"), dict):
            return extract_response_text(data["data"])
        for key in ("text", "content", "output"):
            if isinstance(data.get(key), str):
                return data[key]
    return ""


def build_prompt(meta: dict[str, Any], paper_text: str, extraction_status: str, figures: list[dict[str, Any]] | None = None) -> str:
    authors = ", ".join(meta["authors"])
    source_text = paper_text or meta["summary"]
    figure_lines = "\n".join(f"- {fig['path']} ({fig['source']}, {fig['ext']})" for fig in (figures or [])[:20])
    figure_markdown = build_note_figure_markdown(figures or [], limit=10, heading_level=4)
    return f"""你是一个帮助我打牢大模型基础的论文精读助手。

请基于下面论文信息，输出一份中文 Markdown 笔记。要求极简但有信息密度：

1. 不要写空泛赞美。
2. 重点解释这篇论文解决了什么基础问题。
3. 讲清楚方法主线、关键概念、训练/推理机制、实验结论。
4. 如果只能看到摘要或部分正文，必须明确注明依据有限；如果有图片输入，要结合图片理解方法结构和实验结果。
5. 保留必要英文术语，中文解释要自然。
6. 数学公式使用 Markdown LaTeX。
7. 输出从二级标题开始，不要 YAML frontmatter。
8. 在“关键图表解读”部分必须嵌入原始论文图片，使用下面给出的 Markdown 图片语法，不要只写图片解读。

论文元信息：
- arXiv ID: {meta["arxiv_id"]}
- Title: {meta["title"]}
- Authors: {authors}
- Published: {meta["published"]}
- URL: {meta["abs_url"]}
- Content extraction: {extraction_status}

Abstract:
{meta["summary"]}

Local figure index:
{figure_lines or "No local figures found."}

Markdown figure embeds to include in the note:
{figure_markdown or "No embeddable local figures found."}

Paper/source text excerpt:
{source_text}

请按这个结构输出：

## {meta["title"]}

### 一句话定位
### 基本信息
### 摘要中文翻译
### 研究问题
### 核心方法
### 关键图表解读
### 关键贡献
### 实验与结论
### 局限性
### 放进大模型基础知识体系里怎么理解
### 我需要记住什么
"""


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "paper"


def append_note(config: dict[str, Any], content: str, notes_file: str | None = None) -> Path:
    notes_path = project_root() / (notes_file or config.get("notes_file", "notes/核心论文笔记.md"))
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    if not notes_path.exists():
        notes_path.write_text("# 大模型核心论文笔记\n\n---\n\n", encoding="utf-8")
    with notes_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n---\n\n{content.strip()}\n")
    return notes_path


def write_note_file(content: str, output_dir: str, meta: dict[str, Any]) -> Path:
    output_path = project_root() / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{meta['arxiv_id'].replace('/', '-')}-{slugify(meta['title'])}.md"
    note_path = output_path / filename
    note_path.write_text(content.strip() + "\n", encoding="utf-8")
    return note_path


def write_workspace_note(content: str, paper_dir: Path) -> Path:
    note_path = paper_dir / "note.md"
    note_path.write_text(content.strip() + "\n", encoding="utf-8")
    return note_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, help="arXiv ID, arXiv URL, or search result index.")
    parser.add_argument("--config")
    parser.add_argument("--output-dir", help="Write this paper as one Markdown file under the given project-relative directory.")
    parser.add_argument("--notes-file", help="Append to a project-relative Markdown file. Defaults to config notes_file.")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Fetch metadata/text and print status without calling GPT-5.5.")
    args = parser.parse_args()

    config = load_config(args.config)
    arxiv_id = normalize_arxiv_id(args.paper, config)
    meta = fetch_arxiv_metadata(arxiv_id)
    max_chars = int(config.get("max_input_chars", 28000))

    workspace = None
    if args.output_dir:
        workspace = prepare_paper_workspace(meta, args.output_dir, max_chars, args.no_pdf)
        paper_text = workspace["paper_text"]
        extraction_status = workspace["extraction_status"]
        figures = workspace["figures"]
    else:
        temp_dir = config.get("temp_output_dir", "notes/.tmp")
        workspace = prepare_paper_workspace(meta, temp_dir, max_chars, args.no_pdf)
        paper_text = workspace["paper_text"]
        extraction_status = workspace["extraction_status"]
        figures = workspace["figures"]

    prompt = build_prompt(meta, paper_text, extraction_status, figures)
    if args.dry_run:
        print(f"Title: {meta['title']}")
        print(f"arXiv: {meta['arxiv_id']}")
        print(f"Paper dir: {workspace['paper_dir']}")
        print(f"Content extraction: {extraction_status}")
        print(f"Figures: {len(figures)}")
        print(f"Prompt chars: {len(prompt)}")
        return 0

    max_images = int(config.get("max_images", 6))
    image_parts = image_content_parts(workspace["images_dir"], figures, max_images)
    note = gpt55_request(config, prompt, image_parts)
    note = ensure_note_contains_images(note, build_note_figure_markdown(figures, limit=10, heading_level=4))
    if args.output_dir:
        notes_path = write_workspace_note(note, workspace["paper_dir"])
        print(f"Wrote note: {notes_path}")
    else:
        notes_path = append_note(config, note, args.notes_file)
        print(f"Appended note: {notes_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
