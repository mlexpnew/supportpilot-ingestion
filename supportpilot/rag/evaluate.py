"""Retrieval evaluation for the ReplyPilot knowledge base (SP-123).

Runs every question in the golden set through search and scores where the
first correct article lands.

Two sets (SP-124): ``retrieval_dev.jsonl`` (25) is for tuning and is the
default; ``retrieval_test.jsonl`` (15) is held out and only scored by the CI
gate. Do not look at held-out misses while tuning.

Usage:
    python -m supportpilot.rag.evaluate                      # dev set, grouped
    python -m supportpilot.rag.evaluate --no-group           # raw chunks
    python -m supportpilot.rag.evaluate --json reports/after.json
    python -m supportpilot.rag.evaluate --compare before.json after.json

Metrics (per question, then averaged):
    recall@1  first correct article is ranked 1
    recall@3  a correct article is anywhere in the top 3
    MRR       1 / rank of the first correct article, 0 if not found in depth
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chunker import load_articles

DEFAULT_GOLDEN = Path("data/eval/retrieval_dev.jsonl")  # tune on dev only
HELDOUT_GOLDEN = Path("data/eval/retrieval_test.jsonl")  # CI gate only
DEFAULT_FOLDER = Path("data/help_center")
DEFAULT_DEPTH: int = 10  # how deep we look for MRR (ranks 4-10 still score)
REQUIRED_FIELDS: tuple[str, ...] = (
    "question",
    "expected_article_ids",
    "category",
    "style",
)


class GoldenSetError(ValueError):
    """Raised when the golden set is malformed or points at unknown articles."""


@dataclass(frozen=True)
class GoldenQuestion:
    question: str
    expected_article_ids: tuple[str, ...]
    category: str
    style: str


@dataclass(frozen=True)
class QuestionResult:
    question: str
    expected: list[str]
    category: str
    style: str
    rank: int  # 1-based rank of first correct article, 0 = not found
    returned: list[str]  # article ids in ranked order


@dataclass
class EvalReport:
    group_by_article: bool
    model_name: str
    depth: int
    n_questions: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    recall_at_3_by_category: dict[str, float]
    recall_at_3_by_style: dict[str, float]
    misses: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Golden set
# --------------------------------------------------------------------------


def load_golden(path: str | Path, known_ids: Iterable[str]) -> list[GoldenQuestion]:
    """Load the golden set. Fails loudly on bad lines or unknown article ids."""
    known = set(known_ids)
    questions: list[GoldenQuestion] = []
    unknown: dict[str, list[int]] = defaultdict(list)

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetError(f"{path}:{line_no}: invalid JSON ({exc})") from exc

        missing = [f for f in REQUIRED_FIELDS if not row.get(f)]
        if missing:
            raise GoldenSetError(f"{path}:{line_no}: missing fields {missing}")
        expected = row["expected_article_ids"]
        if not isinstance(expected, list) or not all(
            isinstance(e, str) for e in expected
        ):
            raise GoldenSetError(
                f"{path}:{line_no}: expected_article_ids must be a list of strings"
            )
        for article_id in expected:
            if article_id not in known:
                unknown[article_id].append(line_no)

        questions.append(
            GoldenQuestion(
                question=row["question"],
                expected_article_ids=tuple(expected),
                category=row["category"],
                style=row["style"],
            )
        )

    if unknown:
        detail = ", ".join(f"{a} (lines {v})" for a, v in sorted(unknown.items()))
        raise GoldenSetError(
            f"golden set references article ids not in the knowledge base: {detail}"
        )
    if not questions:
        raise GoldenSetError(f"{path}: golden set is empty")
    return questions


# --------------------------------------------------------------------------
# Scoring (pure functions, no database needed)
# --------------------------------------------------------------------------


def first_correct_rank(returned: Sequence[str], expected: Iterable[str]) -> int:
    """1-based rank of the first returned id that is expected, 0 if none."""
    wanted = set(expected)
    for rank, article_id in enumerate(returned, start=1):
        if article_id in wanted:
            return rank
    return 0


def _hit(rank: int, k: int) -> bool:
    return 0 < rank <= k


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _recall3_by(results: Sequence[QuestionResult], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in results:
        groups[getattr(r, key)].append(1.0 if _hit(r.rank, 3) else 0.0)
    return {name: _mean(vals) for name, vals in sorted(groups.items())}


def score(
    results: Sequence[QuestionResult],
    *,
    group_by_article: bool,
    model_name: str,
    depth: int,
) -> EvalReport:
    """Turn per-question ranks into the overall report."""
    if not results:
        raise ValueError("no results to score")
    misses = [
        {
            "question": r.question,
            "expected": r.expected,
            "got_top3": r.returned[:3],
            "rank": r.rank,
            "category": r.category,
            "style": r.style,
        }
        for r in results
        if not _hit(r.rank, 3)
    ]
    return EvalReport(
        group_by_article=group_by_article,
        model_name=model_name,
        depth=depth,
        n_questions=len(results),
        recall_at_1=_mean([1.0 if r.rank == 1 else 0.0 for r in results]),
        recall_at_3=_mean([1.0 if _hit(r.rank, 3) else 0.0 for r in results]),
        mrr=_mean([1.0 / r.rank if r.rank else 0.0 for r in results]),
        recall_at_3_by_category=_recall3_by(results, "category"),
        recall_at_3_by_style=_recall3_by(results, "style"),
        misses=misses,
        results=[asdict(r) for r in results],
    )


def evaluate_questions(
    questions: Sequence[GoldenQuestion],
    retrieve: Callable[[str], Sequence[str]],
    *,
    group_by_article: bool,
    model_name: str,
    depth: int = DEFAULT_DEPTH,
) -> EvalReport:
    """Score ``retrieve(question) -> ranked article ids`` against the golden set.

    Ranks are slot positions in what search actually returned. With grouping
    off, one article can fill several slots (one per chunk), so a correct
    second article lands lower. That is exactly the "before" behaviour we
    want to measure, so duplicates are kept and never collapsed here.
    """
    results: list[QuestionResult] = []
    for q in questions:
        returned = list(retrieve(q.question))
        results.append(
            QuestionResult(
                question=q.question,
                expected=list(q.expected_article_ids),
                category=q.category,
                style=q.style,
                rank=first_correct_rank(returned, q.expected_article_ids),
                returned=returned,
            )
        )
    return score(
        results, group_by_article=group_by_article, model_name=model_name, depth=depth
    )


# --------------------------------------------------------------------------
# Running against the real store
# --------------------------------------------------------------------------


def run_evaluation(
    golden_path: str | Path = DEFAULT_GOLDEN,
    folder: str | Path = DEFAULT_FOLDER,
    group_by_article: bool = True,
    depth: int = DEFAULT_DEPTH,
    store: Any | None = None,
    embedder: Any | None = None,
) -> EvalReport:
    """Evaluate the live pgvector store with the real embedding model."""
    from .embedder import get_embedder
    from .search import search
    from .store import KBStore, get_kb_engine

    store = store or KBStore(get_kb_engine())
    embedder = embedder or get_embedder()

    article_ids = {a.article_id for a in load_articles(folder)}
    questions = load_golden(golden_path, article_ids)

    indexed = store.article_ids(embedder.model_name)
    not_indexed = sorted(
        {e for q in questions for e in q.expected_article_ids} - indexed
    )
    if not_indexed:
        raise GoldenSetError(
            f"expected articles missing from the store for model "
            f"{embedder.model_name!r}: {not_indexed}. "
            "Run: python -m supportpilot.rag.ingest"
        )

    # Same number of slots either way: grouping off returns ``depth`` raw
    # chunks (duplicates possible), grouping on returns ``depth`` articles.
    def retrieve(question: str) -> list[str]:
        hits = search(
            question, store, embedder, k=depth, group_by_article=group_by_article
        )
        return [h.article_id for h in hits]

    return evaluate_questions(
        questions,
        retrieve,
        group_by_article=group_by_article,
        model_name=embedder.model_name,
        depth=depth,
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def format_report(report: EvalReport) -> str:
    mode = "grouped by article" if report.group_by_article else "raw chunks"
    header = (
        f"Retrieval eval: {report.n_questions} questions, {mode}, "
        f"model={report.model_name}, depth={report.depth}"
    )
    lines = [
        header,
        "",
        f"  recall@1  {report.recall_at_1:.3f}",
        f"  recall@3  {report.recall_at_3:.3f}",
        f"  MRR       {report.mrr:.3f}",
        "",
        "recall@3 by category:",
    ]
    lines += [f"  {c:<14}{v:.3f}" for c, v in report.recall_at_3_by_category.items()]
    lines += ["", "recall@3 by style:"]
    lines += [f"  {s:<14}{v:.3f}" for s, v in report.recall_at_3_by_style.items()]
    lines += ["", f"Misses (not in top 3): {len(report.misses)}"]
    if report.misses:
        lines.append(f"  {'question':<45}{'expected':<38}got (top 3)")
        for m in report.misses:
            lines.append(
                f"  {m['question'][:43]:<45}"
                f"{','.join(m['expected'])[:36]:<38}"
                f"{', '.join(m['got_top3'])}"
            )
    return "\n".join(lines)


def compare_markdown(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Markdown before/after tables for the doc and the PR description."""

    def row(name: str, b: float, a: float) -> str:
        return f"| {name} | {b:.3f} | {a:.3f} | {a - b:+.3f} |"

    out = [
        "| Metric | Before (no grouping) | After (grouped) | Change |",
        "|---|---|---|---|",
        row("recall@1", before["recall_at_1"], after["recall_at_1"]),
        row("recall@3", before["recall_at_3"], after["recall_at_3"]),
        row("MRR", before["mrr"], after["mrr"]),
        "",
        "| Category | recall@3 before | recall@3 after | Change |",
        "|---|---|---|---|",
    ]
    for cat, a in after["recall_at_3_by_category"].items():
        out.append(row(cat, before["recall_at_3_by_category"].get(cat, 0.0), a))
    out += [
        "",
        "| Style | recall@3 before | recall@3 after | Change |",
        "|---|---|---|---|",
    ]
    for style, a in after["recall_at_3_by_style"].items():
        out.append(row(style, before["recall_at_3_by_style"].get(style, 0.0), a))
    out += [
        "",
        f"Misses after grouping ({len(after['misses'])}):",
        "",
        "| Question | Style | Expected | Got (top 3) | Reason |",
        "|---|---|---|---|---|",
    ]
    for m in after["misses"]:
        out.append(
            f"| {m['question']} | {m['style']} | {', '.join(m['expected'])} "
            f"| {', '.join(m['got_top3'])} | TODO |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate KB retrieval (SP-123)")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument(
        "--no-group", action="store_true", help="evaluate raw chunk search"
    )
    parser.add_argument("--json", type=Path, help="also write results to this file")
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BEFORE_JSON", "AFTER_JSON"),
        help="print markdown before/after tables and exit",
    )
    args = parser.parse_args(argv)

    if args.compare:
        before, after = (
            json.loads(p.read_text(encoding="utf-8")) for p in args.compare
        )
        print(compare_markdown(before, after))
        return 0

    report = run_evaluation(
        args.golden, args.folder, group_by_article=not args.no_group, depth=args.depth
    )
    print(format_report(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"\nSaved {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
