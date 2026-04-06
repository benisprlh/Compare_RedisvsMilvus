"""
Load the same peraturan CSV into Milvus (dense HNSW, COSINE) for comparison with Redis.

Uses the same text blobs and OpenAI embeddings as ``load_peraturan_csv.py``.

Requires Milvus reachable at ``MILVUS_URI`` (default ``http://localhost:19530``).

    export OPENAI_API_KEY=sk-...
    export MILVUS_URI=http://localhost:19530
    python3 load_peraturan_milvus.py --limit 500
    python3 load_peraturan_milvus.py   # full ``list.csv/list.csv`` (no --limit)

By default this script **never** calls ``collection.flush()``: only ``insert``, so bulk load
stays fast and data is available for search without waiting on segment seal. Use
``--flush-at-end`` only if you explicitly need a final seal (can block a long time).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

from vector_chat.milvus_store import (
    connect_milvus,
    disconnect_milvus,
    ensure_peraturan_collection,
    flush_collection,
    insert_batch,
)
from vector_chat.peraturan_csv_io import DEFAULT_CSV, iter_csv_chunks, row_to_content

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("load_peraturan_milvus")


def main() -> int:
    p = argparse.ArgumentParser(description="Load peraturan CSV into Milvus.")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to list.csv")
    p.add_argument(
        "--milvus-uri",
        default=os.getenv("MILVUS_URI", "http://localhost:19530"),
        help="Milvus gRPC/http URI (or Lite .db path per pymilvus docs)",
    )
    p.add_argument("--collection", default="peraturan_kb", help="Collection name")
    p.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--chunksize", type=int, default=500)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop collection if it exists before loading",
    )
    p.add_argument(
        "--milvus-timeout",
        type=float,
        default=120.0,
        help="pymilvus gRPC timeout in seconds (default 120)",
    )
    p.add_argument(
        "--flush-at-end",
        action="store_true",
        help="Optional: call flush() once after all inserts (can block a long time; not required to use the data)",
    )
    p.add_argument(
        "--flush-timeout",
        type=float,
        default=7200.0,
        help="Seconds timeout for --flush-at-end only (default 7200)",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        logger.error("CSV not found: %s", args.csv)
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set.")
        return 1

    logger.info("Connecting Milvus at %s (timeout=%ss)", args.milvus_uri, args.milvus_timeout)
    connect_milvus(args.milvus_uri, timeout=args.milvus_timeout)

    try:
        logger.info("OpenAI vectorizer (%s)...", args.embedding_model)
        vectorizer = OpenAITextVectorizer(model=args.embedding_model)
        dims = vectorizer.dims
        logger.info("Embedding dimensions: %s", dims)

        col = ensure_peraturan_collection(
            args.collection,
            dims,
            drop_existing=args.drop_existing,
        )

        row_id = 0
        chunk_idx = 0
        for chunk in iter_csv_chunks(args.csv, args.chunksize, args.limit):
            chunk_idx += 1
            logger.info(
                "CSV chunk %s: parsing up to %s rows (no --limit = full file, can take a long time)...",
                chunk_idx,
                len(chunk),
            )
            ids: List[str] = []
            texts: List[str] = []
            for _, row in chunk.iterrows():
                d: Dict[str, Any] = row.to_dict()
                content = row_to_content(d)
                if not content.strip():
                    continue
                ids.append(f"row-{row_id}")
                row_id += 1
                texts.append(content)

            if not ids:
                continue

            logger.info(
                "Chunk %s: %d documents to embed/insert in batches of %d",
                chunk_idx,
                len(ids),
                args.batch_size,
            )
            batch_num = 0
            for start in range(0, len(ids), args.batch_size):
                batch_num += 1
                sub_ids = ids[start : start + args.batch_size]
                sub_texts = texts[start : start + args.batch_size]
                logger.info(
                    "OpenAI: embedding batch %s (%d texts; API can take tens of seconds)...",
                    batch_num,
                    len(sub_texts),
                )
                vecs = vectorizer.embed_many(
                    sub_texts,
                    batch_size=min(len(sub_texts), 100),
                    as_buffer=False,
                )
                logger.info(
                    "OpenAI: batch %s done; sending to Milvus...",
                    batch_num,
                )
                insert_batch(col, sub_ids, sub_texts, vecs)
                logger.info("Progress: inserted %s .. %s", sub_ids[0], sub_ids[-1])
        if args.flush_at_end:
            flush_collection(col, timeout=args.flush_timeout)
    finally:
        disconnect_milvus()

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
