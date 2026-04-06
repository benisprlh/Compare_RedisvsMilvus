"""Build a RedisVL :class:`~redisvl.index.SearchIndex` for hybrid retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema


def build_chatbot_index_schema(
    name: str = "chatbot_kb",
    prefix: str = "kb",
    vector_dims: int = 384,
    vector_field: str = "embedding",
    content_field: str = "content",
    vector_algorithm: str = "flat",
    vector_attrs: Optional[Dict[str, Any]] = None,
) -> IndexSchema:
    vattrs: Dict[str, Any] = {
        "algorithm": vector_algorithm,
        "dims": vector_dims,
        "distance_metric": "cosine",
        "datatype": "float32",
    }
    if vector_attrs:
        vattrs.update(vector_attrs)
    return IndexSchema.from_dict(
        {
            "version": "0.1.0",
            "index": {
                "name": name,
                "prefix": prefix,
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": [
                {"name": content_field, "type": "text"},
                {"name": vector_field, "type": "vector", "attrs": vattrs},
            ],
        }
    )


def connect_index(redis_url: str, schema: IndexSchema) -> SearchIndex:
    return SearchIndex(schema, redis_url=redis_url)


def seed_records(
    index: SearchIndex,
    records: List[Dict[str, Any]],
    *,
    id_field: str = "id",
) -> List[str]:
    """Load dicts with at least ``id``, ``content``, and ``embedding`` (list[float])."""
    return index.load(records, id_field=id_field)
