from vector_chat.ragas_metrics import compute_context_precision_recall


def test_non_llm_context_metrics():
    question = "What is batik?"
    retrieved = [
        "Batik is Indonesian cultural heritage.",
        "Random unrelated sentence about weather.",
    ]
    gold = ["Batik is Indonesian cultural heritage recognized by UNESCO."]
    cp, cr = compute_context_precision_recall(
        question,
        retrieved,
        reference_answer="unused",
        use_llm=False,
        reference_contexts=gold,
    )
    assert 0 <= cp <= 1
    assert 0 <= cr <= 1
