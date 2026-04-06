from __future__ import annotations

import os
from typing import List, Optional, Tuple

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall
from ragas.metrics import (
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
)


def _llm_from_env():
    from langchain_openai import ChatOpenAI

    model = os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o-mini")
    return LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0))


def compute_context_precision_recall(
    question: str,
    retrieved_contexts: List[str],
    reference_answer: str,
    *,
    use_llm: Optional[bool] = None,
    reference_contexts: Optional[List[str]] = None,
) -> Tuple[float, float]:
    """
    Compute RAGAS ``context_precision`` and ``context_recall``.

    **LLM mode (default when ``OPENAI_API_KEY`` is set):** uses
    :class:`ragas.metrics.ContextPrecision` and :class:`ragas.metrics.ContextRecall`
    with the reference *answer* string (standard RAGAS definitions).

    **Non-LLM fallback:** uses ``NonLLMContextPrecisionWithReference`` and
    ``NonLLMContextRecall`` with ``reference_contexts`` (gold chunk strings).
    Set ``use_llm=False`` to force this path, or leave ``OPENAI_API_KEY`` unset.

    Args:
        question: User query.
        retrieved_contexts: Chunks returned by the retriever (ordered).
        reference_answer: Ground-truth answer (for LLM metrics).
        use_llm: If None, inferred from env.
        reference_contexts: Required for non-LLM mode; gold text chunks.

    Returns:
        (context_precision, context_recall) in [0, 1] or NaN if undefined.
    """
    if use_llm is None:
        use_llm = bool(os.getenv("OPENAI_API_KEY"))

    if use_llm:
        llm = _llm_from_env()
        cp = ContextPrecision(llm=llm)
        cr = ContextRecall(llm=llm)
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
            reference=reference_answer,
        )
        return cp.single_turn_score(sample), cr.single_turn_score(sample)

    if not reference_contexts:
        raise ValueError(
            "reference_contexts is required when use_llm=False (non-LLM RAGAS metrics)."
        )
    cp = NonLLMContextPrecisionWithReference()
    cr = NonLLMContextRecall()
    sample = SingleTurnSample(
        retrieved_contexts=retrieved_contexts,
        reference_contexts=reference_contexts,
    )
    return cp.single_turn_score(sample), cr.single_turn_score(sample)
