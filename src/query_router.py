
from dataclasses import dataclass
import re


PAPER_ALIASES = {
    "1706.03762v7": [
        "attention is all you need",
        "1706.03762",
    ],
    "2410.00037v1": [
        "moshi",
        "2410.00037",
    ],
    "2507.07955v2": [
        "dynamic chunking for end-to-end hierarchical sequence modeling",
        "2507.07955",
        "h-net",
    ],
}


@dataclass
class QueryRoute:
    original_question: str
    search_query: str
    paper_id: str | None
    decision: str


def route_question(question: str) -> QueryRoute:

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty")

    lower_question = question.lower()
    matched_papers = set()

    for paper_id, aliases in PAPER_ALIASES.items():

        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"

            if re.search(pattern, lower_question):
                matched_papers.add(paper_id)

    if len(matched_papers) == 1:
        paper_id = next(iter(matched_papers))
        decision = "explicit_paper"

    elif len(matched_papers) > 1:
        paper_id = None
        decision = "multiple_papers"

    else:
        paper_id = None
        decision = "search_all"

    return QueryRoute(
        original_question=question,
        search_query=question,
        paper_id=paper_id,
        decision=decision,
    )
