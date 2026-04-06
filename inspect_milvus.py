"""Quick peek at a Milvus collection (row count + sample IDs / content preview)."""

from __future__ import annotations

import argparse
import os

from pymilvus import Collection, connections, utility


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect Milvus collection stats and sample rows.")
    p.add_argument(
        "--uri",
        default=os.getenv("MILVUS_URI", "http://localhost:19530"),
    )
    p.add_argument("--collection", default="peraturan_kb")
    p.add_argument("--sample", type=int, default=3, help="How many rows to query (0 = skip)")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    connections.connect(alias="default", uri=args.uri, timeout=args.timeout)
    try:
        if not utility.has_collection(args.collection):
            print(f"No collection named {args.collection!r}.")
            return 1
        col = Collection(args.collection)
        col.load()
        n = col.num_entities
        print(f"collection: {args.collection!r}")
        print(f"num_entities: {n}")
        if args.sample > 0 and n > 0:
            # Primary key is VARCHAR "id"
            out = col.query(
                expr='id != ""',
                output_fields=["id", "content"],
                limit=min(args.sample, max(n, 1)),
            )
            for i, row in enumerate(out, 1):
                cid = row.get("id", "?")
                content = row.get("content", "") or ""
                prev = content[:200] + ("..." if len(content) > 200 else "")
                print(f"\n--- sample {i} id={cid!r} ---\n{prev}")
    finally:
        connections.disconnect("default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
