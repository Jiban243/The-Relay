
"""Manually review The Relay's real-model answers.

Run:
    python -m evaluation.review_quality --question-id a2
    python -m evaluation.review_quality
    python -m evaluation.review_quality --summary

No real model calls are made.
"""

import argparse
import csv
import json
import os

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "quality_full_cca5739647"

GRADES = {
    "c": "correct",
    "w": "wrong",
    "r": "refused_correctly",
    "f": "refused_wrongly",
}


def read_jsonl(path):

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def grade_key(row):
    return (
        row["question_id"],
        row["retrieval_setting"],
    )


def read_grades(path):

    if not path.exists():
        return {}

    return {
        grade_key(item): item
        for item in read_jsonl(path)
    }


def update_csv(csv_path, run_id, grades):

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)
        fields = reader.fieldnames

        if not fields or "answer_grade" not in fields:
            raise ValueError(
                "Required answer_grade column missing"
            )

        rows = list(reader)

    seen = Counter()

    for row in rows:

        if row.get("run_id") != run_id:
            continue

        key = grade_key(row)

        if key in grades:

            seen[key] += 1
            row["answer_grade"] = grades[key]["grade"]

    bad = [
        key
        for key in grades
        if seen[key] != 1
    ]

    if bad:
        raise ValueError(
            "CSV join failed. Expected exactly "
            f"one row per grade: {bad}"
        )

    temporary = csv_path.with_name(
        csv_path.name + ".grading.tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, csv_path)


def pct(values, percentile):

    if not values:
        return "not measured"

    return f"{np.percentile(values, percentile):.2f}"


def report(records, grades):

    print("\n========== QUALITY REVIEW SUMMARY ==========")

    print(
        f"Reviewed {len(grades)}/{len(records)} responses"
    )

    for setting in (
        "small",
        "balanced",
        "reranked",
    ):

        subset = [
            record
            for record in records
            if record["retrieval_setting"] == setting
        ]

        decisions = [
            grades[grade_key(record)]["grade"]
            for record in subset
            if grade_key(record) in grades
        ]

        counts = Counter(decisions)

        traces = [
            record["trace"]
            for record in subset
        ]

        token_totals = [
            trace["tokens_in"] + trace["tokens_out"]
            for trace in traces
            if trace.get("tokens_in") is not None
            and trace.get("tokens_out") is not None
        ]

        first_content = [
            trace["provider_first_content_ms"]
            for trace in traces
            if trace.get("provider_first_content_ms")
            is not None
        ]

        total_times = [
            trace["total_ms"]
            for trace in traces
            if trace.get("total_ms") is not None
        ]

        print(
            f"\n{setting}: "
            f"{len(decisions)}/{len(subset)} reviewed"
        )

        print("Grades:", dict(counts))

        if len(decisions) == len(subset):

            n = len(subset)

            print(
                "Correct and cited:",
                f"{100 * counts['correct'] / n:.1f}%"
            )

            wrong_count = (
                counts["wrong"]
                + counts["refused_wrongly"]
            )

            print(
                "Wrong including wrongful refusals:",
                f"{100 * wrong_count / n:.1f}%"
            )

            print(
                "Correctly refused:",
                f"{100 * counts['refused_correctly'] / n:.1f}%"
            )

        else:
            print(
                "Quality percentages withheld until "
                "all responses are graded."
            )

        if token_totals:
            print(
                "Tokens per answer:",
                f"{sum(token_totals) / len(token_totals):.1f}"
            )

        else:
            print("Tokens per answer: not measured")

        print(
            "Provider first-content p50/p95:",
            pct(first_content, 50),
            "/",
            pct(first_content, 95),
            "ms",
        )

        print(
            "Total time p50/p95:",
            pct(total_times, 50),
            "/",
            pct(total_times, 95),
            "ms",
        )

    print(
        "\nNOTE: Provider first-content latency "
        "is not the same as user-visible TTFT."
    )

    print(
        "The current pipeline buffers answers "
        "until validation finishes."
    )


def display(record):

    trace = record["trace"]

    print("\n" + "=" * 72)

    print(
        record["question_id"],
        "|",
        record["retrieval_setting"],
        "|",
        trace.get("verification"),
    )

    print("\nQuestion:")
    print(record["question"])

    print(
        "\nExpected supported:",
        record["expected_supported"],
    )

    print(
        "Expected paper:",
        record["expected_paper"],
    )

    print("\nDELIVERED ANSWER:")
    print(trace["answer"])

    print("\nRETRIEVED EVIDENCE:")

    for index, item in enumerate(
        record.get("evidence", []),
        start=1,
    ):

        print(
            f"\n{index}. {item['citation']} "
            f"[{item['chunk_id']}]"
        )

        print(
            " ".join(item["text"].split())[:350]
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-id",
        default=RUN_ID,
    )

    parser.add_argument(
        "--question-id",
        default=None,
    )

    parser.add_argument(
        "--summary",
        action="store_true",
    )

    args = parser.parse_args()

    results_dir = ROOT / "results"

    raw_path = (
        results_dir / f"{args.run_id}.jsonl"
    )

    ledger_path = (
        results_dir
        / f"quality_grades_{args.run_id}.jsonl"
    )

    csv_path = results_dir / "results.csv"

    records = read_jsonl(raw_path)

    if not records:
        raise ValueError(
            "No evaluation records found"
        )

    keys = [
        grade_key(record)
        for record in records
    ]

    if len(set(keys)) != len(keys):
        raise ValueError(
            "Duplicate question/setting pairs "
            "in the raw evaluation"
        )

    grades = read_grades(ledger_path)

    # Synchronize any previously saved grades.
    if grades:
        update_csv(
            csv_path,
            args.run_id,
            grades,
        )

    if args.summary:
        report(records, grades)
        return

    pending = [
        record
        for record in records
        if grade_key(record) not in grades
    ]

    if args.question_id:
        pending = [
            record
            for record in pending
            if record["question_id"]
            == args.question_id
        ]

    print("\n========== MANUAL ANSWER REVIEW ==========")

    print("c = correct and cited")
    print("w = wrong")
    print("r = correctly refused")
    print("f = wrongly refused")
    print("e = show full retrieved evidence")
    print("m = show original model response")
    print("s = skip")
    print("q = save progress and quit")

    for record in pending:

        display(record)

        while True:

            action = input(
                "\nGrade (c/w/r/f/e/m/s/q): "
            ).strip().lower()

            if action == "q":
                report(records, grades)
                return

            if action == "s":
                break

            if action == "e":

                for index, item in enumerate(
                    record.get("evidence", []),
                    start=1,
                ):

                    print(
                        f"\nSOURCE {index}: "
                        f"{item['citation']}\n"
                        f"{item['text']}"
                    )

                continue

            if action == "m":

                for index, call in enumerate(
                    record["trace"].get(
                        "model_calls", []
                    ),
                    start=1,
                ):

                    print(
                        f"\nRAW MODEL CALL {index}:\n"
                        f"{call.get('text', '')}"
                    )

                continue

            if action not in GRADES:
                print(
                    "Enter c, w, r, f, e, m, s or q"
                )
                continue

            if (
                action == "c"
                and record["trace"].get(
                    "verification"
                ) != "citation_format_checked"
            ):
                print(
                    "This answer did not pass citation "
                    "validation. It cannot be marked "
                    "correct and cited."
                )

                continue

            note = ""

            if action in ("w", "f"):
                note = input(
                    "Reason for failure: "
                ).strip()

            entry = {
                "reviewed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "run_id": args.run_id,
                "question_id": record["question_id"],
                "retrieval_setting": (
                    record["retrieval_setting"]
                ),
                "grade": GRADES[action],
                "notes": note,
            }

            # Append grading decision to the audit log.
            with ledger_path.open(
                "a",
                encoding="utf-8",
            ) as handle:

                handle.write(
                    json.dumps(
                        entry,
                        ensure_ascii=False,
                    ) + "\n"
                )

            grades[grade_key(record)] = entry

            # Update the corresponding CSV row.
            update_csv(
                csv_path,
                args.run_id,
                grades,
            )

            print(
                "Saved:",
                entry["grade"],
                "-> results.csv + review ledger",
            )

            break

    report(records, grades)

    print("\nReview ledger:", ledger_path)
    print("Updated CSV:", csv_path)


if __name__ == "__main__":
    main()
