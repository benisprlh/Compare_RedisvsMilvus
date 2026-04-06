"""
Retrieve from an existing Redis hybrid index (e.g. after ``load_peraturan_csv.py``)
and print RAGAS ``context_precision`` and ``context_recall``.

Does **not** recreate the index; schema must match how you loaded data.

**LLM mode** (default if ``OPENAI_API_KEY`` is set): needs a reference *answer*
(paraphrase / ground-truth jawaban). Uses chat model from ``RAGAS_OPENAI_MODEL``.

**Non-LLM mode** (``--no-llm``): needs a text file with one *gold* context per line
(strings that should appear in retrieval). No chat LLM calls for the metric.

Examples::

    export REDIS_URL=redis://localhost:6379/0
    export OPENAI_API_KEY=sk-...

    # LLM judges relevance vs reference answer
    python3 eval_peraturan_ragas.py \\
        --query "Peraturan tentang cuti pegawai" \\
        --reference-answer "Cuti pegawai negeri diatur dalam peraturan kepegawaian."

    # String overlap vs gold chunks (no RAGAS chat LLM)
    python3 eval_peraturan_ragas.py --no-llm \\
        --query "..." \\
        --gold-contexts-file gold.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List

from redisvl.index import SearchIndex
from redisvl.utils.vectorize.text.openai import OpenAITextVectorizer

from vector_chat.index_setup import build_chatbot_index_schema, connect_index
from vector_chat.ragas_metrics import compute_context_precision_recall
from vector_chat.service import HybridSearchConfig, RedisHybridRetrievalService


def _read_gold_contexts(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Hybrid retrieve + RAGAS context_precision / context_recall."
    )
    p.add_argument("--query", required=True, help="User question")
    p.add_argument(
        "--reference-answer",
        default="",
        help="Ground-truth answer (required for LLM metric mode)",
    )
    p.add_argument(
        "--gold-contexts-file",
        type=Path,
        help="UTF-8 file: one gold context chunk per line (for --no-llm)",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Use non-LLM RAGAS metrics; requires --gold-contexts-file",
    )
    p.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    p.add_argument("--index-name", default="peraturan_kb")
    p.add_argument("--key-prefix", default="kb")
    p.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--rrf-k", type=int, default=60)
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is required (query embedding uses the same OpenAI model as load).",
            file=sys.stderr,
        )
        return 1

    if not args.no_llm and not args.reference_answer.strip():
        print(
            "LLM mode needs --reference-answer (ground-truth jawaban singkat).",
            file=sys.stderr,
        )
        return 1

    reference_contexts: List[str] | None = None
    if args.no_llm:
        if not args.gold_contexts_file or not args.gold_contexts_file.is_file():
            print(
                "--no-llm requires an existing --gold-contexts-file.",
                file=sys.stderr,
            )
            return 1
        reference_contexts = _read_gold_contexts(args.gold_contexts_file)
        if not reference_contexts:
            print("Gold contexts file is empty.", file=sys.stderr)
            return 1

    vectorizer = OpenAITextVectorizer(model=args.embedding_model)
    dims = vectorizer.dims
    schema = build_chatbot_index_schema(
        name=args.index_name,
        prefix=args.key_prefix,
        vector_dims=dims,
        vector_algorithm="hnsw",
        vector_attrs={"m": 16, "ef_construction": 200},
    )
    index: SearchIndex = connect_index(args.redis_url, schema)

    service = RedisHybridRetrievalService(
        index,
        vectorizer,
        HybridSearchConfig(
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            text_scorer="BM25STD.NORM",
            stopwords="indonesian",
        ),
    )

    print(f"Query: {args.query!r}\n")
    contexts = service.search(args.query)
    for i, c in enumerate(contexts, 1):
        preview = c[:200] + ("..." if len(c) > 200 else "")
        print(f"  [{i}] {preview}\n")

    cp, cr = compute_context_precision_recall(
        args.query,
        contexts,
        reference_answer=args.reference_answer or "(n/a)",
        use_llm=False if args.no_llm else None,
        reference_contexts=reference_contexts,
    )

    def _fmt(x: float) -> str:
        return f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else str(x)

    print(f"context_precision: {_fmt(cp)}")
    print(f"context_recall:    {_fmt(cr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
