"""
Compare RAGAS ``context_precision`` and ``context_recall`` for the **same query**
and **same top_k** between:

- **Redis**: hybrid retrieval (vector + BM25 + RRF) via ``RedisHybridRetrievalService``
- **Milvus**: dense HNSW search, COSINE (same M / efConstruction as Redis vector index)

Use the **same** embedding model and (as far as possible) the **same corpus** as loaded by
``load_peraturan_csv.py`` and ``load_peraturan_milvus.py`` (identical CSV + OpenAI embeddings).

Prereqs: Redis index exists; Milvus collection exists; ``OPENAI_API_KEY`` set.

    export REDIS_URL=redis://localhost:6379/0
    export MILVUS_URI=http://localhost:19530
    export OPENAI_API_KEY=sk-...

    python3 compare_redis_milvus_ragas.py \\
      --query "Peraturan tentang cuti pegawai" \\
      --reference-answer "Cuti pegawai negeri diatur dalam aturan kepegawaian negara." \\
      --top-k 5

    # Several questions (JSON: list of {"query", "reference"})
    python3 compare_redis_milvus_ragas.py --batch compare_eval_batch.json --top-k 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from pymilvus import Collection, utility
from redisvl.index import SearchIndex
from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

from vector_chat.index_setup import build_chatbot_index_schema, connect_index
from vector_chat.milvus_store import connect_milvus, disconnect_milvus, search_contents
from vector_chat.ragas_metrics import compute_context_precision_recall
from vector_chat.service import HybridSearchConfig, RedisHybridRetrievalService


def _read_gold_contexts(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def _fmt_metric(x: float) -> str:
    return f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else str(x)


def _load_batch(path: Path) -> List[Tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("batch JSON must be a list")
    out: List[Tuple[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"item {i} must be an object")
        q = item.get("query", "").strip()
        ref = item.get("reference", "").strip()
        if not q or not ref:
            raise ValueError(f"item {i} needs non-empty query and reference")
        out.append((q, ref))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare RAGAS context metrics: Redis hybrid vs Milvus dense."
    )
    p.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="JSON file: [{\"query\": \"...\", \"reference\": \"...\"}, ...]",
    )
    p.add_argument("--query", default=None)
    p.add_argument("--reference-answer", default="")
    p.add_argument("--gold-contexts-file", type=Path, default=None)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    p.add_argument("--redis-index-name", default="peraturan_kb")
    p.add_argument("--redis-key-prefix", default="kb")
    p.add_argument(
        "--milvus-uri",
        default=os.getenv("MILVUS_URI", "http://localhost:19530"),
    )
    p.add_argument("--milvus-collection", default="peraturan_kb")
    p.add_argument(
        "--milvus-timeout",
        type=float,
        default=120.0,
        help="pymilvus connection timeout (seconds)",
    )
    p.add_argument("--milvus-ef", type=int, default=128, help="HNSW search ef")
    p.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    p.add_argument(
        "--show-contexts",
        action="store_true",
        help="Print retrieved snippets for both backends",
    )
    p.add_argument("--context-preview", type=int, default=180)
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required.", file=sys.stderr)
        return 1

    pairs: List[Tuple[str, str]]
    if args.batch is not None:
        if not args.batch.is_file():
            print(f"--batch file not found: {args.batch}", file=sys.stderr)
            return 1
        try:
            pairs = _load_batch(args.batch)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Invalid --batch JSON: {e}", file=sys.stderr)
            return 1
    else:
        if not args.query or not args.query.strip():
            print("Provide --query and --reference-answer, or --batch FILE.", file=sys.stderr)
            return 1
        pairs = [(args.query.strip(), args.reference_answer.strip())]

    if not args.no_llm:
        for i, (q, ref) in enumerate(pairs):
            if not ref:
                print(f"LLM mode needs reference for item {i}: {q[:60]!r}...", file=sys.stderr)
                return 1

    reference_contexts: Optional[List[str]] = None
    if args.no_llm:
        if not args.gold_contexts_file or not args.gold_contexts_file.is_file():
            print("--no-llm requires --gold-contexts-file.", file=sys.stderr)
            return 1
        reference_contexts = _read_gold_contexts(args.gold_contexts_file)
        if not reference_contexts:
            print("Gold contexts file is empty.", file=sys.stderr)
            return 1

    vectorizer = OpenAITextVectorizer(model=args.embedding_model)
    dims = vectorizer.dims

    schema = build_chatbot_index_schema(
        name=args.redis_index_name,
        prefix=args.redis_key_prefix,
        vector_dims=dims,
        vector_algorithm="hnsw",
        vector_attrs={"m": 16, "ef_construction": 200},
    )
    r_index: SearchIndex = connect_index(args.redis_url, schema)
    r_service = RedisHybridRetrievalService(
        r_index,
        vectorizer,
        HybridSearchConfig(
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            text_scorer="BM25STD.NORM",
            stopwords="indonesian",
        ),
    )

    connect_milvus(args.milvus_uri, timeout=args.milvus_timeout)

    use_llm_flag = False if args.no_llm else None
    sum_cp_r = sum_cr_r = sum_cp_m = sum_cr_m = 0.0
    n_valid = 0

    print(f"top_k={args.top_k}  rrf_k={args.rrf_k}  milvus_ef={args.milvus_ef}")
    print(f"embedding_model={args.embedding_model!r}")
    print(f"items={len(pairs)}")
    print()

    try:
        if not utility.has_collection(args.milvus_collection):
            print(
                f"Milvus collection {args.milvus_collection!r} not found. "
                f"Run: python3 load_peraturan_milvus.py --drop-existing",
                file=sys.stderr,
            )
            return 1
        m_col = Collection(args.milvus_collection)
        m_col.load()

        for idx, (query, ref_ans) in enumerate(pairs, start=1):
            qvec = vectorizer.embed(query, as_buffer=False)
            if not isinstance(qvec, list):
                print("Unexpected embedding type from vectorizer.", file=sys.stderr)
                return 1

            redis_contexts = r_service.search(query)
            milvus_contexts = search_contents(
                m_col, qvec, args.top_k, ef=args.milvus_ef
            )

            cp_r, cr_r = compute_context_precision_recall(
                query,
                redis_contexts,
                reference_answer=ref_ans,
                use_llm=use_llm_flag,
                reference_contexts=reference_contexts,
            )
            cp_m, cr_m = compute_context_precision_recall(
                query,
                milvus_contexts,
                reference_answer=ref_ans,
                use_llm=use_llm_flag,
                reference_contexts=reference_contexts,
            )

            if all(
                isinstance(x, float) and not math.isnan(x)
                for x in (cp_r, cr_r, cp_m, cr_m)
            ):
                sum_cp_r += cp_r
                sum_cr_r += cr_r
                sum_cp_m += cp_m
                sum_cr_m += cr_m
                n_valid += 1

            print(f"=== [{idx}/{len(pairs)}] {query[:100]!r}{'...' if len(query) > 100 else ''} ===")
            if args.show_contexts:
                prev = args.context_preview
                print("--- Redis contexts ---")
                for i, c in enumerate(redis_contexts, 1):
                    pv = c[:prev] + ("..." if len(c) > prev else "")
                    print(f"  [{i}] {pv}\n")
                print("--- Milvus contexts ---")
                for i, c in enumerate(milvus_contexts, 1):
                    pv = c[:prev] + ("..." if len(c) > prev else "")
                    print(f"  [{i}] {pv}\n")

            print(f"{'':20} {'context_precision':>18} {'context_recall':>18}")
            print(f"{'Redis (hybrid)':20} {_fmt_metric(cp_r):>18} {_fmt_metric(cr_r):>18}")
            print(f"{'Milvus (HNSW)':20} {_fmt_metric(cp_m):>18} {_fmt_metric(cr_m):>18}")
            print()

        if len(pairs) > 1 and n_valid > 0:
            print("--- averages (finite scores only) ---")
            print(f"{'':20} {'context_precision':>18} {'context_recall':>18}")
            print(
                f"{'Redis (hybrid)':20} {sum_cp_r / n_valid:>18.4f} {sum_cr_r / n_valid:>18.4f}"
            )
            print(
                f"{'Milvus (HNSW)':20} {sum_cp_m / n_valid:>18.4f} {sum_cr_m / n_valid:>18.4f}"
            )

        return 0
    finally:
        disconnect_milvus()


if __name__ == "__main__":
    raise SystemExit(main())
