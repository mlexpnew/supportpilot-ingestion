"""RAG reply drafting for ReplyPilot (SP-124).

retrieve top-k articles -> build prompt -> LLM draft -> check citations.

Usage:
    python -m supportpilot.rag.draft "my refund hasn't arrived"
    python -m supportpilot.rag.draft "my refund hasn't arrived" --json
    python -m supportpilot.rag.draft --samples data/eval/draft_samples.jsonl \\
        --out docs/replypilot-samples.md

Needs DATABASE_URL (pgvector, ingested) and an LLM chosen by REPLYPILOT_LLM
(see supportpilot/rag/llm.py). Every draft is for a human agent to review.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from .chunker import load_articles
from .llm import LLMClient, LLMError, LLMTimeout, get_llm_client
from .prompt import SYSTEM_PROMPT, PromptArticle, build_user_prompt

DEFAULT_FOLDER = Path("data/help_center")
DEFAULT_K = 3
MAX_TICKET_CHARS = 4000

# A bracket group whose comma-separated parts all look like article ids,
# e.g. [kb-refund-timeline] or [kb-refund-request, kb-refund-timeline].
_ID = r"kb-[a-z0-9]+(?:-[a-z0-9]+)*"
_CITATION_GROUP_RE = re.compile(rf"\[\s*({_ID}(?:\s*,\s*{_ID})*)\s*\]")

# "Here's a draft reply:" style first lines and "Best regards, ..." endings.
_PREAMBLE_RE = re.compile(
    r"^\s*(here(?:'|\u2019)?s|here is|below is|sure[,!]?)[^\n]{0,60}"
    r"(draft|reply|response)[^\n]*:\s*\n+",
    re.IGNORECASE,
)
_SIGNOFF_RE = re.compile(
    r"\n+\s*(best regards|kind regards|warm regards|regards|best|thanks|"
    r"thank you|sincerely|cheers),?\s*(\n[^\n]{0,40}){0,2}\s*$",
    re.IGNORECASE,
)


class Hit(Protocol):
    """What we need from a search result (store.SearchResult fits)."""

    article_id: str
    title: str
    text: str


Retriever = Callable[[str, int], Sequence[Hit]]


@dataclass
class DraftResult:
    draft: str
    citations: list[str]
    retrieved_ids: list[str]
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    invalid_citations: list[str] = field(default_factory=list)
    error: str | None = None  # llm_timeout | llm_error | no_articles

    @property
    def needs_review(self) -> bool:
        """Flag for the agent UI: something about this draft is off."""
        return bool(self.error or self.invalid_citations or not self.citations)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "needs_review": self.needs_review}


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------


def tidy_reply(text: str) -> str:
    """Drop a chatty preface and a signature; the agent adds their own."""
    text = _PREAMBLE_RE.sub("", text, count=1)
    return _SIGNOFF_RE.sub("", text).strip()


def check_citations(
    text: str, allowed_ids: Sequence[str]
) -> tuple[str, list[str], list[str]]:
    """Keep citations to retrieved articles; strip and report made-up ones.

    Returns (cleaned_text, valid_ids_in_order, invalid_ids_in_order).
    """
    allowed = set(allowed_ids)
    valid: list[str] = []
    invalid: list[str] = []

    def fix(match: re.Match[str]) -> str:
        ids = [part.strip() for part in match.group(1).split(",")]
        keep = []
        for article_id in ids:
            bucket = valid if article_id in allowed else invalid
            if article_id not in bucket:
                bucket.append(article_id)
            if article_id in allowed:
                keep.append(article_id)
        return " ".join(f"[{a}]" for a in keep)

    cleaned = _CITATION_GROUP_RE.sub(fix, text)
    cleaned = re.sub(r"[ \t]+([.,!?;:])", r"\1", cleaned)  # "word ." -> "word."
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()
    return cleaned, valid, invalid


# --------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _article_index(folder: str) -> dict[str, PromptArticle]:
    return {
        a.article_id: PromptArticle(a.article_id, a.title, a.body)
        for a in load_articles(folder)
    }


def _default_retriever() -> Retriever:
    from .embedder import get_embedder
    from .search import search
    from .store import KBStore, get_kb_engine

    store = KBStore(get_kb_engine())
    embedder = get_embedder()
    return lambda text, k: search(text, store, embedder, k=k)


def _prompt_articles(hits: Sequence[Hit], folder: Path) -> list[PromptArticle]:
    """Full article text for each retrieved id (fallback: the chunk text)."""
    index = _article_index(str(folder))
    out: list[PromptArticle] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.article_id in seen:
            continue
        seen.add(hit.article_id)
        out.append(
            index.get(hit.article_id)
            or PromptArticle(hit.article_id, hit.title, hit.text)
        )
    return out


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def draft_reply(
    ticket_text: str,
    *,
    llm: LLMClient | None = None,
    retriever: Retriever | None = None,
    k: int = DEFAULT_K,
    folder: str | Path = DEFAULT_FOLDER,
) -> DraftResult:
    """Draft a cited reply for one ticket. Never raises on LLM failures."""
    if not ticket_text or not ticket_text.strip():
        raise ValueError("ticket_text must not be empty")
    if len(ticket_text) > MAX_TICKET_CHARS:
        raise ValueError(f"ticket_text is longer than {MAX_TICKET_CHARS} characters")

    start = time.perf_counter()
    llm = llm or get_llm_client()
    retriever = retriever or _default_retriever()

    def elapsed() -> int:
        return int((time.perf_counter() - start) * 1000)

    articles = _prompt_articles(retriever(ticket_text, k), Path(folder))
    retrieved_ids = [a.article_id for a in articles]
    if not articles:
        return DraftResult("", [], [], llm.model, elapsed(), error="no_articles")

    try:
        resp = llm.generate(SYSTEM_PROMPT, build_user_prompt(ticket_text, articles))
    except LLMTimeout:
        return DraftResult(
            "", [], retrieved_ids, llm.model, elapsed(), error="llm_timeout"
        )
    except LLMError:
        return DraftResult(
            "", [], retrieved_ids, llm.model, elapsed(), error="llm_error"
        )

    draft, citations, invalid = check_citations(tidy_reply(resp.text), retrieved_ids)
    return DraftResult(
        draft=draft,
        citations=citations,
        retrieved_ids=retrieved_ids,
        model=resp.model,
        latency_ms=elapsed(),
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        invalid_citations=invalid,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def format_result(result: DraftResult) -> str:
    lines = [
        result.draft or f"(no draft: {result.error})",
        "",
        f"citations:      {', '.join(result.citations) or '-'}",
        f"retrieved:      {', '.join(result.retrieved_ids) or '-'}",
    ]
    if result.invalid_citations:
        lines.append(f"REMOVED (made up): {', '.join(result.invalid_citations)}")
    lines += [
        f"model:          {result.model}",
        f"latency_ms:     {result.latency_ms}",
        f"tokens in/out:  {result.input_tokens}/{result.output_tokens}",
        f"needs_review:   {result.needs_review}",
    ]
    return "\n".join(lines)


def samples_markdown(
    rows: Sequence[dict[str, Any]], results: Sequence[DraftResult]
) -> str:
    model = results[0].model if results else "-"
    avg_latency = sum(r.latency_ms for r in results) // max(len(results), 1)
    today = datetime.now(UTC).date().isoformat()
    out = [
        "# ReplyPilot draft samples (SP-124)",
        "",
        (
            f"Generated {today} with `{model}`, top-{DEFAULT_K} "
            f"articles, grouped retrieval. Average latency {avg_latency} ms."
        ),
        "",
        (
            "Rating scale: 5 = send as is, 4 = tiny edit, 3 = needs edits, "
            "2 = mostly wrong, 1 = harmful or made up."
        ),
        "",
        "| # | Ticket | Citations | Removed | Rating |",
        "|---|---|---|---|---|",
    ]
    for i, (row, r) in enumerate(zip(rows, results, strict=True), start=1):
        out.append(
            f"| {i} | {row['ticket'][:50]} | {', '.join(r.citations) or '-'} "
            f"| {', '.join(r.invalid_citations) or '-'} |  |"
        )
    out += ["", "**Average rating:** ", ""]
    for i, (row, r) in enumerate(zip(rows, results, strict=True), start=1):
        quoted = "\n".join(f"> {line}" for line in (r.draft or "").splitlines())
        out += [
            f"## {i}. {row.get('title', 'Ticket')}",
            "",
            f"**Ticket:** {row['ticket']}",
            "",
            f"**Expected article(s):** {', '.join(row.get('expected', [])) or '-'}",
            "",
            f"**Retrieved:** {', '.join(r.retrieved_ids) or '-'}",
            "",
            "**Draft:**",
            "",
            quoted or f"> (no draft: {r.error})",
            "",
            f"**Citations:** {', '.join(r.citations) or 'none'}  ",
            f"**Removed (made up):** {', '.join(r.invalid_citations) or 'none'}  ",
            (
                f"**Latency:** {r.latency_ms} ms, tokens {r.input_tokens} in / "
                f"{r.output_tokens} out"
            ),
            "",
            "**Rating (1-5):** ",
            "",
            "**Notes:** ",
            "",
        ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft a cited support reply")
    parser.add_argument("ticket", nargs="?", help="ticket text")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--samples", type=Path, help="JSONL of sample tickets")
    parser.add_argument("--out", type=Path, default=Path("docs/replypilot-samples.md"))
    args = parser.parse_args(argv)
    if not args.ticket and not args.samples:
        parser.error("give a ticket text or --samples FILE")

    llm = get_llm_client()
    retriever = _default_retriever()

    if args.samples:
        rows = [
            json.loads(line)
            for line in args.samples.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        results = []
        for i, row in enumerate(rows, start=1):
            print(f"[{i}/{len(rows)}] {row['ticket'][:60]}")
            results.append(
                draft_reply(row["ticket"], llm=llm, retriever=retriever, k=args.k)
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(samples_markdown(rows, results) + "\n", encoding="utf-8")
        print(f"Saved {args.out}. Now rate each draft by hand.")
        return 0

    result = draft_reply(args.ticket, llm=llm, retriever=retriever, k=args.k)
    print(
        json.dumps(result.to_dict(), indent=2) if args.json else format_result(result)
    )
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
