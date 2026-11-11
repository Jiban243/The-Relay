
import json

import pytest

from src.query_router import route_question
from src.retriever import (
    BM25Retriever,
    RetrievalSetting,
    SETTINGS,
)
from src.evidence import make_grounded_messages


def test_explicit_paper_routing():

    route = route_question(
        "Explain Attention Is All You Need"
    )

    assert route.paper_id == "1706.03762v7"
    assert route.decision == "explicit_paper"


def test_generic_question_searches_all():

    route = route_question(
        "How does chunking work?"
    )

    assert route.paper_id is None
    assert route.decision == "search_all"


def test_multiple_papers():

    route = route_question(
        "Compare Moshi and Attention Is All You Need"
    )

    assert route.paper_id is None
    assert route.decision == "multiple_papers"


def test_empty_question():

    with pytest.raises(ValueError):
        route_question("")


def test_retrieval_settings():

    assert len(SETTINGS) >= 3

    with pytest.raises(ValueError):
        RetrievalSetting(
            name="invalid",
            chunk_words=20,
            top_k=2,
            overlap_words=5,
        )


def test_retrieval_citations(tmp_path):

    records = [
        {
            "paper_id": "1706.03762v7",
            "title": "Attention Is All You Need",
            "page": 3,
            "source_url": (
                "https://arxiv.org/pdf/1706.03762v7"
            ),
            "text": (
                "transformer encoder decoder attention "
                "positional encoding " * 15
            ),
        },
        {
            "paper_id": "2410.00037v1",
            "title": "Moshi",
            "page": 9,
            "source_url": (
                "https://arxiv.org/pdf/2410.00037v1"
            ),
            "text": (
                "speech full duplex audio dialogue " * 15
            ),
        },
    ]

    path = tmp_path / "pages.jsonl"

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    retriever = BM25Retriever(path)

    setting = RetrievalSetting(
        name="test",
        chunk_words=40,
        top_k=3,
        overlap_words=5,
    )

    results = retriever.search(
        "Moshi speech dialogue",
        setting=setting,
    )

    assert results

    for passage in results:
        assert passage.paper_id == "2410.00037v1"
        assert passage.pdf_page == 9
        assert "PDF p. 9" in passage.citation


def test_no_evidence_generation():

    with pytest.raises(ValueError):
        make_grounded_messages(
            question="Unknown question",
            passages=[],
        )
