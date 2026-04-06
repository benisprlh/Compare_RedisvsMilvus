"""Exercise retrieval + RRF without a live Redis server."""

from unittest.mock import MagicMock

from vector_chat.service import HybridSearchConfig, RedisHybridRetrievalService


class FakeVectorizer:
    def embed(self, text: str, *, as_buffer: bool = False, **kwargs):
        out = [0.1, 0.2, 0.3]
        if as_buffer:
            import struct

            return struct.pack(f"{len(out)}f", *out)
        return out


def test_search_fuses_vector_and_text():
    calls = {"n": 0}

    def fake_query(q):
        calls["n"] += 1
        # VectorQuery vs TextQuery: distinguish by class name
        name = type(q).__name__
        if name == "VectorQuery":
            return [{"content": "alpha"}, {"content": "beta"}]
        if name == "TextQuery":
            return [{"content": "beta"}, {"content": "gamma"}]
        raise AssertionError(name)

    index = MagicMock()
    index.query.side_effect = fake_query

    svc = RedisHybridRetrievalService(
        index,
        FakeVectorizer(),
        HybridSearchConfig(top_k=4, rrf_k=60, content_field="content"),
    )
    out = svc.search("hello")
    assert calls["n"] == 2
    # beta wins RRF; then alpha, gamma
    assert out[0] == "beta"
    assert set(out) == {"alpha", "beta", "gamma"}
