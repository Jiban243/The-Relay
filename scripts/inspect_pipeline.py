
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    ROOT / "results" / "pipeline_requests.jsonl"
)

CITE_PATTERN = re.compile(
    r"\[(\d{4}\.\d{4,5}v\d+),\s*PDF p\.\s*(\d+)\]"
)


def main():

    if not RESULTS_FILE.exists():
        print("Pipeline results file not found.")
        return

    records = []

    with RESULTS_FILE.open(
        "r", encoding="utf-8"
    ) as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    real_records = [
        record for record in records
        if record.get("source") == "real"
    ]

    if not real_records:
        print("No real Groq records found.")
        return

    record = real_records[-1]

    print("\n========== PIPELINE DEBUG ==========")

    print("Request ID:", record["request_id"])
    print("Status:", record["status"])
    print("Verification:", record["verification"])

    print("\nRetrieved citations:")

    for citation in record["retrieved_citations"]:
        print(" ", citation)

    print("\n========== RAW MODEL OUTPUT ==========")

    model_calls = record.get("model_calls", [])

    for index, call in enumerate(model_calls, start=1):

        print(f"\nModel call {index}")
        print("Status:", call.get("status"))
        print("Finish reason:", call.get("finish_reason"))

        text = call.get("text") or ""

        print("\nGenerated answer:")
        print(text)

        citations = CITE_PATTERN.findall(text)

        print("\nRecognized citations:")
        print(citations)

        if not citations:
            print(
                "No citations matched our required format."
            )

    print("\n========== FINAL OUTPUT ==========")
    print(record["answer"])


if __name__ == "__main__":
    main()
