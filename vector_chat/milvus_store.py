"""Milvus collection for peraturan: dense vectors, COSINE, HNSW (aligned with Redis vector index)."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


def _pymilvus_dense_only_skip_broken_system_scipy() -> None:
    """
    pymilvus calls ``SciPyHelper._init()`` on insert to detect sparse matrices; that
    executes ``scipy.sparse``. Debian/Ubuntu ``python3-scipy`` is often incompatible
    with pip's NumPy 2.x (``numpy.core.multiarray failed to import``). This project
    only uses **dense** float vectors, so we mark the helper as already initialized
    with no sparse support and avoid importing SciPy entirely.
    """
    from pymilvus.client import utils as _pv_utils

    _pv_utils.SciPyHelper._checked = True


_pymilvus_dense_only_skip_broken_system_scipy()

# VARCHAR max length for content (chars); matches indexed text cap order of magnitude
_CONTENT_MAX_LEN = 65535
logger = logging.getLogger(__name__)


def connect_milvus(
    uri: str, alias: str = "default", *, timeout: Optional[float] = None
) -> None:
    """
    Connect pymilvus. Use ``http://host:19530`` for server or Milvus Lite ``.db`` URI.

    ``timeout`` is in seconds (gRPC); default 120 to avoid apparent hangs on slow ops.
    """
    kw = {"alias": alias, "uri": uri}
    if timeout is not None:
        kw["timeout"] = timeout
    else:
        kw["timeout"] = 120.0
    connections.connect(**kw)


def disconnect_milvus(alias: str = "default") -> None:
    connections.disconnect(alias)


def ensure_peraturan_collection(
    collection_name: str,
    dim: int,
    *,
    drop_existing: bool = False,
    alias: str = "default",
) -> Collection:
    """
    Create (or reopen) collection ``id`` (VARCHAR), ``content`` (VARCHAR), ``embedding`` (FLOAT_VECTOR).

    Index: HNSW, COSINE, M=16, efConstruction=200 (same HNSW hyperparameters as Redis load script).
    """
    if drop_existing and utility.has_collection(collection_name, using=alias):
        utility.drop_collection(collection_name, using=alias)

    if utility.has_collection(collection_name, using=alias):
        logger.info("Milvus: opening existing collection %r (load may take a while)...", collection_name)
        col = Collection(collection_name, using=alias)
        col.load()
        logger.info("Milvus: collection %r loaded", collection_name)
        return col

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=256,
            auto_id=False,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=_CONTENT_MAX_LEN,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="peraturan kb")
    logger.info("Milvus: creating collection %r (dim=%s)...", collection_name, dim)
    col = Collection(collection_name, schema, using=alias)
    logger.info("Milvus: building HNSW index...")
    col.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    logger.info("Milvus: loading collection into memory...")
    col.load()
    logger.info("Milvus: collection %r ready", collection_name)
    return col


def insert_batch(
    collection: Collection,
    ids: Sequence[str],
    contents: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    *,
    flush: bool = False,
    flush_timeout: Optional[float] = None,
) -> None:
    """
    Insert one batch. By default **no** ``flush()`` — Milvus ``flush`` seals segments and
    can block a long time per call; for bulk loads, call :func:`flush_collection` once at
    the end instead.
    """
    if not ids:
        return
    n = len(ids)
    logger.info("Milvus: inserting %d rows (gRPC)...", n)
    collection.insert([list(ids), list(contents), list(embeddings)])
    logger.info("Milvus: insert RPC done (%d rows buffered)", n)
    if flush:
        flush_collection(collection, timeout=flush_timeout)


def flush_collection(collection: Collection, *, timeout: Optional[float] = 3600.0) -> None:
    """
    Seal all segments for the collection. Use a finite ``timeout`` (seconds) so the client
    does not wait forever if the server stalls (default 3600s).
    """
    logger.info(
        "Milvus: flush (sealing segments; timeout=%ss) — can take minutes on large data...",
        timeout,
    )
    collection.flush(timeout=timeout)
    logger.info("Milvus: flush finished")


def search_contents(
    collection: Collection,
    query_vector: List[float],
    top_k: int,
    *,
    ef: int = 128,
) -> List[str]:
    """Dense ANN search; returns ``content`` strings in score order (best first)."""
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": ef}},
        limit=top_k,
        output_fields=["content"],
    )
    out: List[str] = []
    for hit in results[0]:
        # pymilvus Hit: hit.get("content") resolves output field from entity
        c = hit.get("content")
        if isinstance(c, str):
            out.append(c)
    return out
