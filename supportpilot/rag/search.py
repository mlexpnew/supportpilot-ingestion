"""Semantic search over the knowledge base (SP-122).

Usage:
    python -m supportpilot.rag.search "where is my parcel" [-k 5]
"""

from __future__ import annotations

import argparse

from .embedder import Embedder, get_embedder
from .store import KBStore, SearchResult, get_kb_engine


def search(
    query: str, store: KBStore, embedder: Embedder, k: int = 5
) -> list[SearchResult]:
    """Embed the query the same way as the chunks, then rank by cosine."""
    if not query.strip():
        raise ValueError("query must not be empty")
    return store.search(embedder.embed_query(query), embedder.model_name, k=k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the knowledge base")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    results = search(args.query, KBStore(get_kb_engine()), get_embedder(), k=args.k)
    if not results:
        print("No results. Did you run: python -m supportpilot.rag.ingest ?")
        return 1
    for rank, r in enumerate(results, start=1):
        snippet = r.text.split("\n", 1)[-1].replace("\n", " ")[:90]
        print(f"{rank}. {r.score:.3f}  {r.title}  [{r.article_id}]  {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
