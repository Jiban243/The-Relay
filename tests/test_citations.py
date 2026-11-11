
from types import SimpleNamespace

from src.pipeline import (
    check_citations,
    extract_citations,
    normalize_citations,
)


PASSAGES = [
    SimpleNamespace(
        paper_id="2410.00037v1",
        pdf_page=6,
    ),
    SimpleNamespace(
        paper_id="2410.00037v1",
        pdf_page=5,
    ),
]


def test_standard_square_brackets():
    answer = "Moshi uses two streams [2410.00037v1, PDF p. 6]"
    assert check_citations(answer, PASSAGES)


def test_unicode_citation_brackets():
    answer = "Moshi uses two streams 【2410.00037v1, PDF p. 6】"
    assert check_citations(answer, PASSAGES)


def test_original_groq_answer():
    answer = (
        "Moshi achieves full-duplex dialogue "
        "using semantic and acoustic tokens "
        "【2410.00037v1, PDF p. 6】"
        "【2410.00037v1, PDF p. 5】"
    )

    assert check_citations(answer, PASSAGES)

    normalized = normalize_citations(answer)

    assert "【" not in normalized
    assert "[2410.00037v1, PDF p. 6]" in normalized
    assert "[2410.00037v1, PDF p. 5]" in normalized


def test_non_retrieved_page_is_rejected():
    answer = "Claim 【2410.00037v1, PDF p. 99】"
    assert not check_citations(answer, PASSAGES)


def test_missing_citation_is_rejected():
    assert not check_citations(
        "Moshi uses semantic and acoustic tokens.",
        PASSAGES,
    )


def test_wrong_paper_is_rejected():
    answer = "Claim [1706.03762v7, PDF p. 6]"
    assert not check_citations(answer, PASSAGES)


def test_mismatched_brackets_are_rejected():
    answer = "Claim 【2410.00037v1, PDF p. 6]"
    assert not check_citations(answer, PASSAGES)


def test_extract_both_citation_styles():
    answer = (
        "[2410.00037v1, PDF p. 5] "
        "【2410.00037v1, PDF p. 6】"
    )

    assert extract_citations(answer) == [
        ("2410.00037v1", 5),
        ("2410.00037v1", 6),
    ]
