"""Reciprocal Rank Fusion (RRF) for merging ranked text lists.

redis-vl (redisvl) does not ship this helper in all versions; this matches the
pattern: ``_, fused = reciprocal_rank_fusion(vec_list, text_list, k=rrf_k)``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    vector_ranked: List[str],
    text_ranked: List[str],
    k: int = 60,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Fuse two ordered lists of document texts using RRF.

    score(d) = sum_i 1 / (k + rank_i(d)) across lists where d appears.

    Args:
        vector_ranked: Contents in vector search order (best first).
        text_ranked: Contents in full-text search order (best first).
        k: RRF smoothing constant (same role as Redis hybrid ``rrf_constant``).

    Returns:
        Tuple of (per-document RRF scores, fused ranking by descending score).
    """
    scores: Dict[str, float] = {}

    for rank, doc in enumerate(vector_ranked, start=1):
        if not doc:
            continue
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    for rank, doc in enumerate(text_ranked, start=1):
        if not doc:
            continue
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return scores, fused
