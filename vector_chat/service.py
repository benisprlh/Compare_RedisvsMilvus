from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Union

from redisvl.index import SearchIndex
from redisvl.query import TextQuery, VectorQuery

from vector_chat.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class SupportsEmbed(Protocol):
    def embed(
        self, text: str, *, as_buffer: bool = False, **kwargs: Any
    ) -> Union[List[float], bytes]: ...


@dataclass
class HybridSearchConfig:
    """Field names and retrieval hyperparameters."""

    vector_field: str = "embedding"
    content_field: str = "content"
    top_k: int = 5
    rrf_k: int = 60
    text_scorer: str = "BM25STD.NORM"
    stopwords: str = "indonesian"


class RedisHybridRetrievalService:
    """
    Chatbot-oriented hybrid retrieval: dense vector + BM25 full-text, fused with RRF.

    Mirrors the flow you specified (VectorQuery + TextQuery + ``reciprocal_rank_fusion``).
    """

    def __init__(
        self,
        index: SearchIndex,
        vectorizer: SupportsEmbed,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        self.index = index
        self.vectorizer = vectorizer
        self.config = config or HybridSearchConfig()

    def search(self, query: str) -> List[str]:
        """Return up to ``top_k`` fused context strings for ``query``."""
        cfg = self.config
        query_vector = self.vectorizer.embed(query, as_buffer=True)

        vector_query = VectorQuery(
            vector=query_vector,
            vector_field_name=cfg.vector_field,
            num_results=cfg.top_k,
            return_fields=[cfg.content_field],
            return_score=False,
        )
        vector_results = self.index.query(vector_query)
        vector_contents = [r[cfg.content_field] for r in vector_results]

        text_contents: List[str] = []
        try:
            text_query = TextQuery(
                text=query,
                text_field_name=cfg.content_field,
                text_scorer=cfg.text_scorer,
                num_results=cfg.top_k,
                return_fields=[cfg.content_field],
                stopwords=cfg.stopwords,
                return_score=False,
            )
            text_results = self.index.query(text_query)
            text_contents = [r[cfg.content_field] for r in text_results]
        except ValueError as e:
            logger.warning(
                "Full-text query skipped (%s). Using vector-only list for fusion.", e
            )
            text_contents = []

        _, fused_results = reciprocal_rank_fusion(
            vector_contents, text_contents, k=cfg.rrf_k
        )
        return fused_results[: cfg.top_k]
