
"""Regression tests for paper-scoped retrieval."""

from pathlib import Path
from types import SimpleNamespace

from src.pipeline import select_paper_scope
from src.query_router import route_question
from src.retriever import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]

TRANSFORMER = "1706.03762v7"
MOSHI = "2410.00037v1"
HNET = "2507.07955v2"


def test_original_transformer_alias():

    question = (
        "What are the encoder and decoder "
        "components of the Transformer?"
    )

    paper_id, reason = select_paper_scope(
        question
    )

    assert paper_id == TRANSFORMER
    assert reason in (
        "explicit_paper",
        "transformer_alias",
    )


def test_explicit_paper_has_priority():

    explicit_route = SimpleNamespace(
        paper_id=MOSHI
    )

    paper_id, reason = select_paper_scope(
        "How does Moshi work?",
        explicit_route,
    )

    assert paper_id == MOSHI
    assert reason == "explicit_paper"


def test_hnet_mention_not_misrouted():

    no_explicit_route = SimpleNamespace(
        paper_id=None
    )

    paper_id, reason = select_paper_scope(
        "How does a Transformer work inside H-Net?",
        no_explicit_route,
    )

    assert paper_id is None
    assert reason == "all_three_papers"


def test_generic_question_searches_all():

    no_explicit_route = SimpleNamespace(
        paper_id=None
    )

    paper_id, reason = select_paper_scope(
        "Compare the three research papers.",
        no_explicit_route,
    )

    assert paper_id is None
    assert reason == "all_three_papers"


def test_a2_retrieval_uses_transformer_paper():

    question = (
        "What are the encoder and decoder "
        "components of the Transformer?"
    )

    route = route_question(question)

    paper_id, _ = select_paper_scope(
        question,
        route,
    )

    retriever = BM25Retriever(
        ROOT / "data/processed/pages.jsonl"
    )

    for setting in (
        "small",
        "balanced",
        "reranked",
    ):

        passages = retriever.search(
            question,
            setting=setting,
            paper_id=paper_id,
        )

        assert passages, (
            f"No retrieved passages: {setting}"
        )

        assert all(
            passage.paper_id == TRANSFORMER
            for passage in passages
        ), (
            f"Wrong paper retrieved: {setting}"
        )
