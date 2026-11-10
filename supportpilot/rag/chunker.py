"""Help-centre article loader and heading-aware chunker (SP-122).

Articles live in ``data/help_center/*.md`` and start with front-matter::

    ---
    id: kb-track-order
    title: How to track your order
    category: delivery
    updated_at: 2026-11-10
    ---

The body is split at ``##`` headings. Sections longer than ``max_words`` are
split again into overlapping windows so no chunk exceeds the limit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS: tuple[str, ...] = ("id", "title", "category", "updated_at")
ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {"delivery", "refund", "billing", "cancellation", "account", "other"}
)
DEFAULT_MAX_WORDS: int = 300
DEFAULT_OVERLAP_WORDS: int = 50

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


class ArticleFormatError(ValueError):
    """Raised when an article file is missing front-matter or required fields."""


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    category: str
    updated_at: str
    body: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    article_id: str
    title: str
    category: str
    chunk_index: int
    text: str
    content_hash: str


def content_hash(text: str) -> str:
    """Stable SHA-256 hex digest of chunk text (used for idempotent ingest)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_article(raw: str, source_path: str = "<memory>") -> Article:
    """Parse one markdown article with front-matter. Fails loudly on bad input."""
    match = _FRONT_MATTER_RE.match(raw.replace("\r\n", "\n"))
    if match is None:
        raise ArticleFormatError(f"{source_path}: missing '---' front-matter block")

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ArticleFormatError(f"{source_path}: bad front-matter line {line!r}")
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()

    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise ArticleFormatError(f"{source_path}: missing fields {missing}")
    if meta["category"] not in ALLOWED_CATEGORIES:
        raise ArticleFormatError(
            f"{source_path}: unknown category {meta['category']!r}"
        )

    body = match.group(2).strip()
    if not body:
        raise ArticleFormatError(f"{source_path}: empty body")

    return Article(
        article_id=meta["id"],
        title=meta["title"],
        category=meta["category"],
        updated_at=meta["updated_at"],
        body=body,
        source_path=source_path,
    )


def load_articles(folder: str | Path) -> list[Article]:
    """Load every ``.md`` article in ``folder`` (sorted by filename)."""
    folder_path = Path(folder)
    paths = sorted(folder_path.glob("*.md"))
    if not paths:
        raise ArticleFormatError(f"no .md articles found in {folder_path}")

    articles = [parse_article(p.read_text(encoding="utf-8"), str(p)) for p in paths]

    seen: set[str] = set()
    for article in articles:
        if article.article_id in seen:
            raise ArticleFormatError(f"duplicate article id {article.article_id!r}")
        seen.add(article.article_id)
    return articles


def split_sections(body: str) -> list[str]:
    """Split a body at ``##`` headings, keeping each heading with its text."""
    sections: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [s for s in sections if s]


def _window_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap_words
    windows: list[str] = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return windows


def chunk_article(
    article: Article,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Turn an article into chunks of at most ``max_words`` words.

    Each chunk is prefixed with the article title so it still makes sense on
    its own (and embeds with the right topic).
    """
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words")

    chunks: list[Chunk] = []
    for section in split_sections(article.body):
        for piece in _window_words(section, max_words, overlap_words):
            text = f"{article.title}\n{piece}"
            chunks.append(
                Chunk(
                    article_id=article.article_id,
                    title=article.title,
                    category=article.category,
                    chunk_index=len(chunks),
                    text=text,
                    content_hash=content_hash(text),
                )
            )
    return chunks


def chunk_folder(
    folder: str | Path,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Load and chunk every article in ``folder``."""
    return [
        chunk
        for article in load_articles(folder)
        for chunk in chunk_article(article, max_words, overlap_words)
    ]
