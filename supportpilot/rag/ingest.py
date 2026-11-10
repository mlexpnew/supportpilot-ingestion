"""Ingest help-centre articles into the pgvector knowledge base (SP-122).

Usage:
    python -m supportpilot.rag.ingest [--folder data/help_center]

Idempotent: only new or changed chunks are embedded; removed or edited chunks
are deleted. Running it twice in a row adds nothing the second time.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .chunker import chunk_folder
from .embedder import Embedder, get_embedder
from .store import KBStore, SyncResult, get_kb_engine

logger = logging.getLogger(__name__)

DEFAULT_FOLDER = Path("data/help_center")


def ingest_folder(folder: str | Path, store: KBStore, embedder: Embedder) -> SyncResult:
    """Chunk, embed only what is new, and sync the store."""
    chunks = chunk_folder(folder)

    unique = {c.content_hash: c for c in chunks}  # drop exact duplicate chunks
    existing = store.existing_hashes(embedder.model_name)
    new_chunks = [c for h, c in unique.items() if h not in existing]

    vectors = embedder.embed_texts([c.text for c in new_chunks])
    return store.apply(new_chunks, vectors, unique.keys(), embedder.model_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest help-centre articles")
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    store = KBStore(get_kb_engine())
    result = ingest_folder(args.folder, store, get_embedder())
    print(
        f"added={result.added} unchanged={result.unchanged} "
        f"removed={result.removed} total={result.total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
