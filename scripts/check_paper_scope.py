
"""Check the Transformer retrieval fix without using Groq."""

from pathlib import Path

from src.pipeline import select_paper_scope
from src.query_router import route_question
from src.retriever import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]

QUESTION = (
    "What are the encoder and decoder "
    "components of the Transformer?"
)


def main():

    retriever = BM25Retriever(
        ROOT / "data/processed/pages.jsonl"
    )

    route = route_question(QUESTION)

    paper_id, reason = select_paper_scope(
        QUESTION,
        route,
    )

    print("\n========== PAPER ROUTING CHECK ==========")

    print("Question:", QUESTION)
    print("Original route:", route.decision)
    print("Selected paper:", paper_id)
    print("Scope reason:", reason)

    for setting in (
        "small",
        "balanced",
        "reranked",
    ):

        passages = retriever.search(
            QUESTION,
            setting=setting,
            paper_id=paper_id,
        )

        print(f"\nSetting: {setting}")

        for passage in passages:

            print(
                "Citation:",
                passage.citation,
            )

            print(
                "Text:",
                " ".join(
                    passage.text.split()
                )[:350],
            )

            print()

    print(
        "\nNo real Groq requests were made."
    )


if __name__ == "__main__":
    main()
