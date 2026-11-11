
import argparse
from pathlib import Path

from src.query_router import route_question
from src.retriever import BM25Retriever, SETTINGS


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        choices=SETTINGS.keys(),
        default="balanced",
    )

    parser.add_argument(
        "--question",
        default=(
            "What are the encoder and decoder "
            "components of the Transformer?"
        ),
    )

    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]

    pages_path = (
        ROOT / "data" / "processed" / "pages.jsonl"
    )

    retriever = BM25Retriever(pages_path)

    route = route_question(args.question)

    results = retriever.search(
        question=args.question,
        setting=args.setting,
    )

    print("\n========== RETRIEVAL TEST ==========")

    print(f"\nQuestion: {args.question}")

    print(f"Routing decision: {route.decision}")

    print(f"Selected paper: {route.paper_id}")

    print(f"Retrieval setting: {args.setting}")

    print(
        f"Total indexed chunks: "
        f"{retriever.chunk_count(args.setting)}"
    )

    print(f"Retrieved passages: {len(results)}")

    for i, passage in enumerate(results, start=1):

        preview = " ".join(
            passage.text.split()
        )[:300]

        print(f"\n--- Result {i} ---")

        print(f"Paper: {passage.title}")

        print(f"Citation: {passage.citation}")

        print(f"BM25 score: {passage.score:.4f}")

        print(f"Preview: {preview}...")

    if not results:
        print("\nNo matching evidence found.")

    print("\nRetrieval test complete.")


if __name__ == "__main__":
    main()
