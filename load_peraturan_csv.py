"""
Flush the current Redis logical database, recreate the search index, embed rows
from ``list.csv/list.csv``, and load them for hybrid search.

**Security:** set ``OPENAI_API_KEY`` in your environment. Do not put API keys in
source code or commit them to git. If a key was pasted in chat, rotate it.

Usage::

    set OPENAI_API_KEY=sk-...   # PowerShell: $env:OPENAI_API_KEY = \"...\"
    set REDIS_URL=redis://localhost:6379/0
    python load_peraturan_csv.py
    python load_peraturan_csv.py --limit 500   # smoke test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import redis
from redisvl.index import SearchIndex
from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

from vector_chat.index_setup import build_chatbot_index_schema, seed_records
from vector_chat.peraturan_csv_io import DEFAULT_CSV, iter_csv_chunks, row_to_content

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("load_peraturan_csv")


def main() -> int:
    p = argparse.ArgumentParser(description="Load peraturan CSV into Redis with OpenAI embeddings.")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to list.csv")
    p.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    p.add_argument("--index-name", default="peraturan_kb", help="RediSearch index name")
    p.add_argument("--key-prefix", default="kb", help="Redis key prefix for documents")
    p.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    p.add_argument("--batch-size", type=int, default=100, help="Rows per OpenAI embed + Redis load batch")
    p.add_argument("--chunksize", type=int, default=500, help="Pandas read_csv chunk size")
    p.add_argument("--limit", type=int, default=None, help="Max rows to index (for testing)")
    p.add_argument("--no-flushdb", action="store_true", help="Do not FLUSHDB (use with care)")
    args = p.parse_args()

    if not args.csv.is_file():
        logger.error("CSV not found: %s", args.csv)
        return 1

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OPENAI_API_KEY is not set. Set it in your environment (do not commit it to git)."
        )
        return 1

    logger.info("Initializing OpenAI vectorizer (%s)...", args.embedding_model)
    vectorizer = OpenAITextVectorizer(model=args.embedding_model)
    dims = vectorizer.dims
    logger.info("Embedding dimensions: %s", dims)

    schema = build_chatbot_index_schema(
        name=args.index_name,
        prefix=args.key_prefix,
        vector_dims=dims,
        vector_algorithm="hnsw",
        vector_attrs={"m": 16, "ef_construction": 200},
    )
    index = SearchIndex(schema, redis_url=args.redis_url)

    raw = redis.from_url(args.redis_url, decode_responses=False)
    if not args.no_flushdb:
        logger.warning("FLUSHDB on %s — all keys in this logical DB will be deleted.", args.redis_url)
        raw.flushdb()

    logger.info("Creating search index %s (overwrite + drop keys)...", args.index_name)
    index.create(overwrite=True, drop=True)

    row_id = 0
    for chunk in iter_csv_chunks(args.csv, args.chunksize, args.limit):
        records: List[Dict[str, Any]] = []
        texts: List[str] = []
        for _, row in chunk.iterrows():
            d = row.to_dict()
            content = row_to_content(d)
            if not content.strip():
                continue
            rid = f"row-{row_id}"
            row_id += 1
            texts.append(content)
            records.append({"id": rid, "content": content})

        if not records:
            continue

        for start in range(0, len(records), args.batch_size):
            sub_recs = records[start : start + args.batch_size]
            sub_texts = texts[start : start + args.batch_size]
            vecs = vectorizer.embed_many(
                sub_texts,
                batch_size=min(len(sub_texts), 100),
                as_buffer=True,
            )
            for rec, emb in zip(sub_recs, vecs):
                rec["embedding"] = emb
            seed_records(index, sub_recs, id_field="id")
            logger.info(
                "Indexed ids %s .. %s",
                sub_recs[0]["id"],
                sub_recs[-1]["id"],
            )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
