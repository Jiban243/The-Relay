
"""Offline quality audit for The Relay.

Uses the original 45-response JSONL.
Preserves existing manual grades.
Makes no API calls.
"""

import csv
import json
import os
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RUN_ID = "quality_full_cca5739647"

CSV_PATH = RESULTS / "results.csv"
RAW_PATH = RESULTS / f"{RUN_ID}.jsonl"

LEDGER_PATH = (
    RESULTS / f"quality_grades_{RUN_ID}.jsonl"
)

AUDIT_PATH = RESULTS / "QUALITY_GRADE_AUDIT.jsonl"
SUMMARY_PATH = RESULTS / "QUALITY_GRADE_SUMMARY.md"


# Explicit concept-coverage rubrics.
# Each pattern is required, with alternative wording.
# These are screening criteria, not proof that
# every additional statement is factually correct.

RUBRICS = {
    "a1": [
        r"softmax",
        r"(?:sqrt|square root|scale|divid)",
        r"(?:value|weighted)",
    ],
    "a2": [
        r"encoder",
        r"decoder",
        r"(?:attention|feed.forward)",
    ],
    "a3": [
        r"(?:position|order)",
        r"(?:encoding|embedding)",
        r"(?:recurr|convol|sequence)",
    ],
    "a4": [
        r"(?:multi.head|heads?)",
        r"(?:parallel|multiple|several)",
        r"(?:project|linear)",
        r"(?:concat|combin)",
    ],
    "m1": [
        r"(?:full.duplex|listen|speaking|speak)",
        r"(?:stream|simultaneous|parallel|overlap|real.time)",
        r"(?:speech|audio)",
    ],
    "m2": [
        r"mimi",
        r"(?:codec|tokeniz|encod|quantiz)",
        r"(?:audio|speech)",
    ],
    "m3": [
        r"semantic",
        r"acoustic",
        r"token",
    ],
    "m4": [
        r"(?:inner.monologue|text)",
        r"text",
        r"(?:speech|audio|token)",
    ],
    "h1": [
        r"chunk",
        r"(?:dynamic|learn|context)",
        r"(?:compress|down.sampl|segment|boundar|split)",
    ],
    "h2": [
        r"(?:chunk|boundar)",
        r"(?:routing|router|similarity|gating)",
        r"(?:learn|train|end.to.end|predict|optim)",
    ],
    "h3": [
        r"(?:routing|router)",
        r"(?:boundar|chunk|split)",
        r"(?:decid|probabil|similarity|predict|determin)",
    ],
    "h4": [
        r"encoder",
        r"main",
        r"decoder",
        r"(?:compress|down.sampl|hierarch|reconstruct|up.sampl|process)",
    ],
}


EXPECTED_PAPERS = {
    **{
        q: "1706.03762v7"
        for q in ("a1", "a2", "a3", "a4")
    },
    **{
        q: "2410.00037v1"
        for q in ("m1", "m2", "m3", "m4")
    },
    **{
        q: "2507.07955v2"
        for q in ("h1", "h2", "h3", "h4")
    },
    "u1": None,
    "u2": None,
    "u3": None,
}

VALID_GRADES = {
    "correct",
    "wrong",
    "refused_correctly",
    "refused_wrongly",
}

CITATION = re.compile(
    r"[\[【]\s*"
    r"(\d{4}\.\d{4,5}v\d+)\s*,\s*"
    r"PDF\s*p\.\s*(\d+)\s*"
    r"[\]】]",
    re.IGNORECASE,
)


def normalize(text):
    text = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    )

    text = re.sub(
        r"[\u2010-\u2015\u2212]",
        "-",
        text,
    )

    return text.casefold()


def is_refusal(answer):

    answer = normalize(answer).lstrip()

    return answer.startswith((
        "i cannot verify",
        "i can't verify",
        "the supplied papers do not",
        "the provided papers do not",
        "the supplied passages do not",
        "the provided passages do not",
        "the papers do not",
        "this information is not present",
        "insufficient evidence",
        "there is insufficient evidence",
        "the excerpts do not",
    ))


def read_jsonl(path):

    with path.open(encoding="utf-8") as file:

        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def classify(record):

    question_id = record["question_id"]

    expected_paper = EXPECTED_PAPERS[
        question_id
    ]

    trace = record["trace"]
    answer = trace.get("answer", "")

    cited = {
        (paper, int(page))
        for paper, page in CITATION.findall(answer)
    }

    retrieved = set()

    for item in record.get("evidence", []):

        retrieved.update({
            (paper, int(page))
            for paper, page in CITATION.findall(
                item.get("citation", "")
            )
        })

    # Questions absent from the three-paper library
    # should receive an honest refusal.
    if is_refusal(answer):

        if expected_paper is None:

            return (
                "refused_correctly",
                ["Correctly refused an out-of-library question"],
                "high",
            )

        return (
            "refused_wrongly",
            ["Refused a question answerable from the papers"],
            "high",
        )

    if expected_paper is None:

        return (
            "wrong",
            ["Answered an out-of-library question"],
            "high",
        )

    notes = []

    if trace.get("status") != "ok":
        notes.append(
            "Model request did not complete successfully"
        )

    if not cited:
        notes.append(
            "No recognized source citation"
        )

    if cited and not cited.issubset(retrieved):
        notes.append(
            "Citation references a non-retrieved page"
        )

    if not any(
        paper == expected_paper
        for paper, _ in cited
    ):
        notes.append(
            "No citation to the expected research paper"
        )

    if (
        trace.get("verification")
        != "citation_format_checked"
    ):
        notes.append(
            "Pipeline did not pass citation validation"
        )

    normalized_answer = normalize(answer)

    missing = [
        pattern
        for pattern in RUBRICS[question_id]
        if not re.search(
            pattern,
            normalized_answer,
            re.IGNORECASE,
        )
    ]

    if missing:
        notes.append(
            "Required concepts missing: "
            + ", ".join(missing)
        )

    if notes:

        return "wrong", notes, "medium"

    return (
        "correct",
        [
            "Core concepts and citation provenance passed "
            "the offline rubric; additional unsupported "
            "claims are not independently ruled out"
        ],
        "medium",
    )


def main():

    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    records = read_jsonl(RAW_PATH)

    keys = [
        (
            record["question_id"],
            record["retrieval_setting"],
        )
        for record in records
    ]

    if len(records) != 45 or len(set(keys)) != 45:
        raise RuntimeError(
            "Expected exactly 45 unique evaluations"
        )

    if {
        record["question_id"]
        for record in records
    } != set(EXPECTED_PAPERS):
        raise RuntimeError(
            "Unexpected evaluation question IDs"
        )

    # Read existing human decisions.
    # These always take priority.
    manual = {}

    if LEDGER_PATH.exists():

        for entry in read_jsonl(LEDGER_PATH):

            if entry.get("grade") in VALID_GRADES:

                key = (
                    entry["question_id"],
                    entry["retrieval_setting"],
                )

                manual[key] = entry["grade"]

    grades = {}
    audit = []

    for record in records:

        key = (
            record["question_id"],
            record["retrieval_setting"],
        )

        prediction, reasons, confidence = (
            classify(record)
        )

        chosen = manual.get(
            key,
            prediction,
        )

        grades[key] = chosen

        audit.append({
            "run_id": RUN_ID,
            "question_id": key[0],
            "retrieval_setting": key[1],
            "expected_paper": (
                EXPECTED_PAPERS[key[0]]
            ),
            "grade": chosen,
            "method": (
                "human_existing"
                if key in manual
                else "offline_rubric_v1"
            ),
            "rubric_prediction": prediction,
            "confidence": (
                "human"
                if key in manual
                else confidence
            ),
            "notes": reasons,
        })

    # Read existing CSV, preserving every other run.

    with CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        fields = reader.fieldnames
        rows = list(reader)

    if (
        not fields
        or "answer_grade" not in fields
    ):
        raise RuntimeError(
            "CSV is missing answer_grade"
        )

    seen = Counter()
    filled = 0

    for row in rows:

        if row.get("run_id") != RUN_ID:
            continue

        key = (
            row.get("question_id"),
            row.get("retrieval_setting"),
        )

        if key not in grades:
            raise RuntimeError(
                f"Unexpected quality row: {key}"
            )

        seen[key] += 1

        if row.get("answer_grade"):

            if row["answer_grade"] != grades[key]:

                raise RuntimeError(
                    "Existing grade differs. "
                    f"Refusing to overwrite: {key}"
                )

        else:

            row["answer_grade"] = grades[key]
            filled += 1

    if any(
        seen[key] != 1
        for key in grades
    ):
        raise RuntimeError(
            "CSV and evaluation JSONL do not "
            "match one-to-one"
        )

    # Back up the original CSV before changing it.

    backup = (
        RESULTS / "results.before_offline_grading.csv"
    )

    if filled and not backup.exists():
        shutil.copy2(
            CSV_PATH,
            backup,
        )

    if filled:

        temp = CSV_PATH.with_suffix(
            ".grading.tmp"
        )

        with temp.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(rows)

        os.replace(
            temp,
            CSV_PATH,
        )

    # Save one auditable decision per response.

    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for item in audit:

            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                ) + "\n"
            )

    # Generate quality table.

    lines = [
        "# The Relay — Quality Audit",
        "",
        "Method: existing human grades preserved; "
        "remaining responses screened using a frozen "
        "offline answer-concept and citation rubric.",
        "",
        "This is not independent expert verification "
        "of every factual claim.",
        "",
        "| Setting | N | Correct and cited | Wrong incl. wrongful refusal | Correctly refused | Avg. tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for setting in (
        "small",
        "balanced",
        "reranked",
    ):

        subset = [
            item
            for item in audit
            if item["retrieval_setting"] == setting
        ]

        counts = Counter(
            item["grade"]
            for item in subset
        )

        originals = [
            record
            for record in records
            if record["retrieval_setting"] == setting
        ]

        token_values = []

        for record in originals:

            trace = record["trace"]

            if (
                trace.get("tokens_in") is not None
                and trace.get("tokens_out") is not None
            ):

                token_values.append(
                    int(trace["tokens_in"])
                    + int(trace["tokens_out"])
                )

        avg_tokens = (
            sum(token_values) / len(token_values)
            if token_values
            else None
        )

        token_text = (
            f"{avg_tokens:.1f}"
            if avg_tokens is not None
            else "not measured"
        )

        lines.append(
            f"| {setting} | 15 | "
            f"{counts['correct'] / 15 * 100:.1f}% | "
            f"{(counts['wrong'] + counts['refused_wrongly']) / 15 * 100:.1f}% | "
            f"{counts['refused_correctly'] / 15 * 100:.1f}% | "
            f"{token_text} |"
        )

    lines.extend([
        "",
        "## Individual decisions",
        "",
        "| Question | Setting | Grade | Method | Reason |",
        "|---|---|---|---|---|",
    ])

    for item in audit:

        explanation = "; ".join(
            item["notes"]
        ).replace("|", "/")

        lines.append(
            f"| {item['question_id']} | "
            f"{item['retrieval_setting']} | "
            f"{item['grade']} | "
            f"{item['method']} | "
            f"{explanation} |"
        )

    lines.extend([
        "",
        "Limitation: lexical concept checks can miss "
        "unsupported additional claims and can reject "
        "valid paraphrases. The scores are rubric-based.",
    ])

    SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n========== QUALITY AUDIT ==========")

    print(
        "Original evaluation records:",
        len(records),
    )

    print(
        "Existing human grades preserved:",
        len(manual),
    )

    print(
        "New offline rubric grades written:",
        filled,
    )

    for setting in (
        "small",
        "balanced",
        "reranked",
    ):

        counts = Counter(
            item["grade"]
            for item in audit
            if item["retrieval_setting"] == setting
        )

        print(setting, dict(counts))

    print("\nSaved:", AUDIT_PATH)
    print("Saved:", SUMMARY_PATH)
    print("Updated:", CSV_PATH)

    if backup.exists():
        print("Original CSV backup:", backup)


if __name__ == "__main__":
    main()
