"""Tests for the help-centre loader and chunker (SP-122)."""

from __future__ import annotations

from pathlib import Path

import pytest

from supportpilot.rag.chunker import (
    ALLOWED_CATEGORIES,
    ArticleFormatError,
    chunk_article,
    chunk_folder,
    content_hash,
    load_articles,
    parse_article,
    split_sections,
)

HELP_CENTER = Path(__file__).resolve().parents[1] / "data" / "help_center"

VALID = """---
id: kb-test
title: Test article
category: delivery
updated_at: 2026-11-10
---

## First
Alpha beta gamma.

## Second
Delta epsilon.
"""


def test_all_help_center_articles_load() -> None:
    articles = load_articles(HELP_CENTER)
    assert len(articles) == 18
    assert {a.category for a in articles} == ALLOWED_CATEGORIES
    assert len({a.article_id for a in articles}) == 18


def test_every_category_has_at_least_two_articles() -> None:
    articles = load_articles(HELP_CENTER)
    for category in ALLOWED_CATEGORIES:
        assert sum(a.category == category for a in articles) >= 2


@pytest.mark.parametrize("field", ["id", "title", "category", "updated_at"])
def test_missing_field_raises(field: str) -> None:
    raw = "\n".join(
        line for line in VALID.splitlines() if not line.startswith(f"{field}:")
    )
    with pytest.raises(ArticleFormatError, match=field):
        parse_article(raw, "bad.md")


def test_missing_front_matter_raises() -> None:
    with pytest.raises(ArticleFormatError, match="front-matter"):
        parse_article("## Just a body\nNo header here.", "bad.md")


def test_unknown_category_raises() -> None:
    with pytest.raises(ArticleFormatError, match="category"):
        parse_article(VALID.replace("category: delivery", "category: shoes"))


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(VALID, encoding="utf-8")
    (tmp_path / "b.md").write_text(VALID, encoding="utf-8")
    with pytest.raises(ArticleFormatError, match="duplicate"):
        load_articles(tmp_path)


def test_split_sections_keeps_headings() -> None:
    sections = split_sections(parse_article(VALID).body)
    assert len(sections) == 2
    assert sections[0].startswith("## First")
    assert sections[1].startswith("## Second")


def test_chunks_carry_metadata_and_title() -> None:
    chunks = chunk_article(parse_article(VALID))
    assert [c.chunk_index for c in chunks] == [0, 1]
    for c in chunks:
        assert c.article_id == "kb-test"
        assert c.category == "delivery"
        assert c.text.startswith("Test article\n")
        assert c.content_hash == content_hash(c.text)


def test_long_section_split_with_overlap() -> None:
    words = [f"w{i}" for i in range(700)]
    raw = VALID.replace("Alpha beta gamma.", " ".join(words))
    chunks = chunk_article(parse_article(raw), max_words=300, overlap_words=50)
    # "## First" + 700 words = 702 words -> windows [0:300], [250:550], [500:702]
    long_chunks = chunks[:-1]  # the last chunk is the short "## Second" section
    assert len(long_chunks) == 3
    assert chunks[-1].text.endswith("Delta epsilon.")
    first_words = long_chunks[0].text.split()
    second_words = long_chunks[1].text.split()
    assert first_words[-50:] == second_words[2 : 2 + 50]  # skip 2 title words
    assert long_chunks[-1].text.split()[-1] == "w699"


def test_no_chunk_exceeds_limit_on_real_articles() -> None:
    for c in chunk_folder(HELP_CENTER):
        body_words = c.text.split("\n", 1)[1].split()
        assert len(body_words) <= 300


def test_hash_is_stable() -> None:
    assert content_hash("same text") == content_hash("same text")
    assert content_hash("same text") != content_hash("other text")


def test_chunking_is_deterministic() -> None:
    first = [c.content_hash for c in chunk_folder(HELP_CENTER)]
    second = [c.content_hash for c in chunk_folder(HELP_CENTER)]
    assert first == second


def test_bad_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_article(parse_article(VALID), max_words=50, overlap_words=50)
