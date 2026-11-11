
import json
from pathlib import Path
from types import SimpleNamespace

from src.pipeline import (
    CITE_PATTERN,
    check_citations,
    normalize_citations,
)


ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    ROOT / "results" / "pipeline_requests.jsonl"
)


def main():

    with RESULTS_FILE.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    real_records = [
        record
        for record in records
        if record.get("source") == "real"
        and record.get("model_calls")
    ]

    if not real_records:
        print("No saved real-model response found.")
        return

    record = real_records[-1]

    answer = record["model_calls"][-1].get("text", "")

    passages = []

    for citation in record["retrieved_citations"]:

        match = CITE_PATTERN.search(citation)

        if match:
            passages.append(
                SimpleNamespace(
                    paper_id=match.group("paper"),
                    pdf_page=int(match.group("page")),
                )
            )

    valid = check_citations(answer, passages)

    print("\n========== CITATION RECHECK ==========")
    print("Request ID:", record["request_id"])
    print("Original verification:", record["verification"])
    print("Citation format valid:", valid)

    print("\nOriginal model answer:")
    print(answer)

    if valid:
        print("\nNormalized accepted answer:")
        print(normalize_citations(answer))
    else:
        print("\nCitation validation still failed.")

    print(
        "\nNote: This checks source locations, "
        "not factual correctness."
    )


if __name__ == "__main__":
    main()
