"""
End-to-end demo: multilingual embeddings, Redis hybrid index, RRF retrieval, optional RAGAS.

Requires Redis with RediSearch + vector (e.g. Redis Stack). Start with:

    docker compose up -d

Then:

    set REDIS_URL=redis://localhost:6379/0
    python run_demo.py
"""

from __future__ import annotations

import os
import sys

from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

from vector_chat.index_setup import build_chatbot_index_schema, connect_index, seed_records
from vector_chat.ragas_metrics import compute_context_precision_recall
from vector_chat.service import HybridSearchConfig, RedisHybridRetrievalService

# Sample Indonesian KB (tourism / policy style snippets)
SAMPLE_DOCS = [
    {
        "id": "1",
        "content": "Batik adalah warisan budaya Indonesia yang diakui UNESCO sejak 2009.",
    },
    {
        "id": "2",
        "content": "Komodo adalah spesies kadal besar yang hanya ditemukan di Pulau Komodo Indonesia.",
    },
    {
        "id": "3",
        "content": "Rendang berasal dari Minangkabau dan sering disebut sebagai salah satu makanan terenak di dunia.",
    },
    {
        "id": "4",
        "content": "Candi Borobudur adalah monumen Buddha terbesar di dunia dan terletak di Magelang.",
    },
]


def main() -> int:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    print("Loading embedding model (first run may download weights)...")
    vectorizer = HFTextVectorizer(model=model, dtype="float32")
    dims = vectorizer.dims
    print(f"Vector dims: {dims}")

    schema = build_chatbot_index_schema(vector_dims=dims)
    index = connect_index(redis_url, schema)

    print(f"Using Redis at {redis_url} (lazy connect on first operation)...")
    try:
        index.create(overwrite=True, drop=True)
    except Exception as e:
        print(
            "\nCould not reach Redis or create the search index.\n"
            f"Details: {e}\n\n"
            "- Use Redis Stack (RediSearch + vector), e.g.:\n"
            "    docker compose up -d\n"
            "- Point REDIS_URL to your instance if needed.\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    records = []
    for doc in SAMPLE_DOCS:
        emb = vectorizer.embed(doc["content"], as_buffer=True)
        records.append(
            {
                "id": doc["id"],
                "content": doc["content"],
                "embedding": emb,
            }
        )
    seed_records(index, records)
    print(f"Indexed {len(records)} documents.")

    service = RedisHybridRetrievalService(
        index,
        vectorizer,
        HybridSearchConfig(
            top_k=3,
            rrf_k=60,
            text_scorer="BM25STD.NORM",
            stopwords="indonesian",
        ),
    )

    query = os.getenv("DEMO_QUERY", "Apa itu warisan budaya batik?")
    print(f"\nQuery: {query!r}")
    contexts = service.search(query)
    for i, c in enumerate(contexts, 1):
        print(f"  [{i}] {c[:120]}{'...' if len(c) > 120 else ''}")

    # RAGAS: non-LLM if no API key (uses gold chunk list)
    gold = [SAMPLE_DOCS[0]["content"]]
    try:
        cp, cr = compute_context_precision_recall(
            query,
            contexts,
            reference_answer="Batik adalah warisan budaya Indonesia yang diakui UNESCO.",
            reference_contexts=gold,
        )
        print(f"\nRAGAS context_precision: {cp:.4f}")
        print(f"RAGAS context_recall: {cr:.4f}")
    except Exception as e:
        print(f"\nRAGAS metrics skipped or failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
