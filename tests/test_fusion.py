from vector_chat.fusion import reciprocal_rank_fusion


def test_rrf_orders_by_combined_rank():
    v = ["a", "b", "c"]
    t = ["b", "d", "a"]
    scores, fused = reciprocal_rank_fusion(v, t, k=60)
    assert set(scores.keys()) == {"a", "b", "c", "d"}
    # "b" appears high in both lists -> should be first
    assert fused[0] == "b"
