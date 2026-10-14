"""Turn web pages and files into clean, section-aware chunks with stable IDs.

Every source is first converted to Markdown, so one chunker handles all formats:
  * Web page  → the site's clean .md copy if it has one, else the main HTML article
  * .md/.txt  → as is
  * .html     → main article only (no menus, sidebars, footers)
  * .docx     → Word heading styles become # headings
  * .pdf      → one "## Page N" section per page, so citations show the page number
Then: split by headings → split by size → add a "Document / Section" header → hash ID.
"""

import hashlib
import ipaddress
import logging
import re
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx", ".html", ".htm"}
HTTP_HEADERS = {"User-Agent": "AgenticRAG-Copilot/2.0"}
MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
BOILERPLATE_RE = re.compile(
    r"(llms\.txt|Documentation Index|Skip to main content|Copy page|Was this page helpful"
    r"|Edit this page|Get your tickets)",
    re.IGNORECASE,
)


class IngestionError(ValueError):
    """A document could not be loaded or produced no usable text."""


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------


def title_from_markdown(md: str) -> Optional[str]:
    match = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", md, re.M) or re.search(r"^#\s+(.+)$", md, re.M)
    return match.group(1).strip() if match else None


def clean_markdown(md: str) -> str:
    md = md.replace("\r\n", "\n")
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)  # drop YAML front matter
    lines = [line for line in md.splitlines() if not BOILERPLATE_RE.search(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_markdown(root) -> str:
    """Small HTML → Markdown converter: keeps headings, paragraphs, list items and code."""
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"]):
        if el.find_parent(["pre", "li"]):  # already captured by its parent block
            continue
        if el.name == "pre":
            blocks.append("```\n" + el.get_text().strip() + "\n```")
        elif el.name in ("h1", "h2", "h3", "h4"):
            blocks.append("#" * int(el.name[1]) + " " + el.get_text(" ", strip=True))
        else:
            text = el.get_text(" ", strip=True)
            if text:
                blocks.append(("- " if el.name == "li" else "") + text)
    return "\n\n".join(blocks)


def main_html_to_markdown(html: str, fallback_title: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else fallback_title
    main = soup.find("article") or soup.find("main") or soup.body or soup
    for tag in main.find_all(["nav", "aside", "header", "footer", "script", "style", "button", "svg", "form"]):
        tag.decompose()
    return clean_markdown(html_to_markdown(main)), title


# ---------------------------------------------------------------------------
# Loaders: every loader returns (markdown, title)
# ---------------------------------------------------------------------------


def check_public_url(url: str) -> None:
    """Block URLs that point inside your network (SSRF protection).

    A server that fetches any URL it is given can be tricked into reading internal services
    (cloud metadata endpoints, admin panels on 10.x/192.168.x). Only public addresses pass.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestionError("Only http(s) URLs with a host name can be ingested.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise IngestionError(f"Cannot resolve host {parsed.hostname}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise IngestionError(f"{parsed.hostname} resolves to a private address; refusing to fetch it.")


def load_url(url: str, timeout: int = 20, allow_private: bool = False) -> tuple[str, str]:
    if allow_private:
        if not url.startswith(("http://", "https://")):
            raise IngestionError("Only http(s) URLs can be ingested.")
    else:
        check_public_url(url)
    # 1) Clean Markdown copy of the page, if the site provides one.
    try:
        resp = requests.get(url.rstrip("/") + ".md", headers=HTTP_HEADERS, timeout=timeout)
        if resp.ok and not resp.text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            return clean_markdown(resp.text), title_from_markdown(resp.text) or url
    except requests.RequestException as exc:
        logger.info("No Markdown copy for %s (%s)", url, exc)
    # 2) The HTML page, main article only.
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise IngestionError(f"Could not download {url}: {exc}") from exc
    return main_html_to_markdown(resp.text, url)


def _load_docx(path: Path) -> str:
    import docx

    lines = []
    for para in docx.Document(str(path)).paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        level = re.match(r"heading (\d)", style)
        if style == "title":
            lines.append(f"# {text}")
        elif level:  # Title is level 1, so "Heading 1" becomes ##, "Heading 2" becomes ###
            lines.append("#" * min(int(level.group(1)) + 1, 3) + f" {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    pages = []
    for number, page in enumerate(PdfReader(str(path)).pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"## Page {number}\n\n{text}")
    return "\n\n".join(pages)


def load_file(path: Path, display_name: Optional[str] = None) -> tuple[str, str]:
    """Load a local file as (markdown, title). `display_name` is the name shown in citations."""
    name = display_name or path.name
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    try:
        if suffix in {".md", ".markdown", ".txt"}:
            raw = path.read_text(encoding="utf-8", errors="replace")
            md, title = clean_markdown(raw), title_from_markdown(raw)
        elif suffix in {".html", ".htm"}:
            md, title = main_html_to_markdown(path.read_text(encoding="utf-8", errors="replace"), name)
        elif suffix == ".docx":
            md = clean_markdown(_load_docx(path))
            title = title_from_markdown(md)
        else:  # .pdf
            md, title = clean_markdown(_load_pdf(path)), None
    except IngestionError:
        raise
    except Exception as exc:  # corrupt or encrypted files
        raise IngestionError(f"Could not read {name}: {exc}") from exc
    if not md.strip():
        raise IngestionError(f"{name} contains no extractable text (scanned PDFs need OCR first).")
    return md, title or Path(name).stem.replace("_", " ").replace("-", " ").strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_markdown(md: str, source: str, title: str, settings: Settings) -> list[Document]:
    """Split by headings, then by size; prefix each chunk with title + section path."""
    sections = MarkdownHeaderTextSplitter(MD_HEADERS, strip_headers=False).split_text(md)
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    chunks: dict[str, Document] = {}  # keyed by ID → automatic de-duplication
    for section in sections:
        path = " > ".join(section.metadata[h] for _, h in MD_HEADERS if h in section.metadata)
        for piece in splitter.split_text(section.page_content):
            piece = piece.strip()
            if len(piece) < settings.min_chunk_chars:
                continue
            header = f"Document: {title}" + (f"\nSection: {path}" if path else "")
            chunk_id = hashlib.sha256(f"{source}|{path}|{piece}".encode()).hexdigest()[:32]
            chunks[chunk_id] = Document(
                id=chunk_id,
                page_content=f"{header}\n\n{piece}",
                metadata={"chunk_id": chunk_id, "source": source, "title": title, "section": path, "source_type": "kb"},
            )
    return list(chunks.values())


def safe_filename(name: str) -> str:
    """Strip folders and unusual characters from an uploaded file's name."""
    base = Path(name or "upload").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._") or "upload"
    return f"{stem[:80]}{Path(base).suffix.lower()}"
