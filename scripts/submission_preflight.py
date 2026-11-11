
"""The Relay — final submission preflight.

Run:
    python -m scripts.submission_preflight

Read-only: does not call APIs or modify source data.
"""

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "SUBMISSION_WRITEUP.md"

QUALITY_RUN = "quality_full_cca5739647"

REQUIRED_COLUMNS = {
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
}

HEADINGS = [
    "## 1. What I built and why",
    "## 2. The real service and my stand-in",
    "## 3. Results",
    "## 4. Where the time goes",
    "## 5. When a limit is hit",
    "## Retrieval: quality versus cost",
    "## How this library will mislead a naive retriever",
    "## 6. Retries",
    "## 7. What the person sees",
    "## 8. What I'm unsure about, what I left out, what I learned",
    "## Your feedback on Vaani",
    "## How I worked with the assistant",
]

findings = []


def result(level, message):
    findings.append((level, message))
    print(f"[{level}] {message}")


def check_files():

    required = [
        "SUBMISSION_WRITEUP.md",
        "results/results.csv",
        "results/FINAL_MEASUREMENT_AUDIT.json",
        "results/QUALITY_GRADE_AUDIT.jsonl",
        f"results/{QUALITY_RUN}.jsonl",
        "src/pipeline.py",
        "src/retriever.py",
        "src/standin_service.py",
        "src/admission.py",
        "src/retry_policy.py",
    ]

    for name in required:
        if (ROOT / name).is_file():
            result("PASS", f"Found {name}")
        else:
            result("FAIL", f"Missing {name}")

    paper_dir = ROOT / "data/papers"

    pdfs = list(paper_dir.rglob("*.pdf")) if paper_dir.exists() else []

    if len(pdfs) >= 3:
        result("PASS", f"Found {len(pdfs)} paper PDFs")
    else:
        result("WARN", "Fewer than three paper PDFs found")


def check_csv():

    path = RESULTS / "results.csv"

    if not path.exists():
        return

    with path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        rows = list(reader)

    missing = REQUIRED_COLUMNS - fields

    if missing:
        result("FAIL", f"Missing CSV columns: {sorted(missing)}")
    else:
        result("PASS", "All mandatory CSV columns present")

    ids = [row.get("request_id") for row in rows]

    if len(set(ids)) == len(ids):
        result("PASS", f"{len(rows)} unique request IDs")
    else:
        result("FAIL", "Duplicate request IDs detected")

    statuses = {"ok", "rate_limited", "timeout", "error"}
    sources = {"real", "stand-in"}

    invalid_status = sum(
        row.get("status") not in statuses
        for row in rows
    )

    invalid_source = sum(
        row.get("source") not in sources
        for row in rows
    )

    if invalid_status or invalid_source:
        result(
            "FAIL",
            f"Invalid statuses: {invalid_status}; "
            f"invalid sources: {invalid_source}",
        )
    else:
        result("PASS", "Statuses and source labels valid")

    quality = [
        row for row in rows
        if row.get("run_id") == QUALITY_RUN
    ]

    if len(quality) != 45:
        result(
            "FAIL",
            f"Expected 45 quality records; found {len(quality)}",
        )
        return

    grades = Counter(
        row.get("answer_grade") for row in quality
    )

    if "" in grades or None in grades:
        result("FAIL", "Some quality grades are missing")
    else:
        result("PASS", "All 45 quality grades present")

    for setting in ("small", "balanced", "reranked"):
        subset = [
            row for row in quality
            if row["retrieval_setting"] == setting
        ]

        counts = Counter(
            row["answer_grade"] for row in subset
        )

        result(
            "INFO",
            f"{setting}: {len(subset)} questions, "
            f"grades={dict(counts)}",
        )

    audit_path = RESULTS / "QUALITY_GRADE_AUDIT.jsonl"

    if audit_path.exists():
        with audit_path.open(encoding="utf-8") as stream:
            audit = [
                json.loads(line)
                for line in stream
                if line.strip()
            ]

        methods = Counter(
            row.get("method") for row in audit
        )

        if len(audit) == 45:
            result(
                "PASS",
                f"Grading provenance recorded: {dict(methods)}",
            )
        else:
            result(
                "FAIL",
                f"Expected 45 grading audit entries; found {len(audit)}",
            )


def check_report():

    if not REPORT.exists():
        return

    text = REPORT.read_text(encoding="utf-8")

    positions = [text.find(heading) for heading in HEADINGS]

    if all(position >= 0 for position in positions):
        result("PASS", "All 12 required headings found")
    else:
        missing = [
            heading for heading, position
            in zip(HEADINGS, positions)
            if position < 0
        ]
        result("FAIL", f"Missing headings: {missing}")

    if positions == sorted(positions) and min(positions) >= 0:
        result("PASS", "Required heading order correct")
    else:
        result("FAIL", "Required heading order incorrect")

    for table in ("Table 1", "Table 2a", "Table 2b", "Table 3"):
        if table in text:
            result("PASS", f"{table} present")
        else:
            result("FAIL", f"{table} missing")

    words = len(re.findall(r"\b[\w-]+\b", text))

    result("INFO", f"Markdown word count: {words}")

    if words > 1200:
        result(
            "WARN",
            "High risk of exceeding two pages; "
            "verify the rendered PDF page count",
        )

    if "**Table 2a" in text and "**Table 2b" in text:

        section = text.split(
            "**Table 2a", 1
        )[1].split(
            "**Table 2b", 1
        )[0]

        rows = [
            line for line in section.splitlines()
            if line.startswith("|")
            and not re.match(r"^\|\s*---", line)
        ]

        missing_duration = []

        for line in rows[1:]:
            cells = [
                part.strip()
                for part in line.strip("|").split("|")
            ]

            if len(cells) >= 4 and cells[3] == "not measured":
                missing_duration.append(cells[0])

        if missing_duration:
            result(
                "WARN",
                "Scenario duration missing in Table 2a: "
                + ", ".join(missing_duration),
            )
        else:
            result("PASS", "Table 2a duration cells populated")

    result(
        "WARN",
        "Two-page PDF pagination and first-person "
        "website feedback require final verification",
    )

    result(
        "WARN",
        "Attach the actual, unedited assistant "
        "conversation history separately",
    )


def check_git():

    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        result("WARN", "Git safety check unavailable")
        return

    if completed.returncode != 0:
        result("WARN", "Not inside an initialized Git repository")
        return

    tracked = [
        path.strip()
        for path in completed.stdout.splitlines()
        if path.strip()
    ]

    dangerous = [
        path for path in tracked
        if (
            Path(path).name == ".env"
            or path.startswith(".venv/")
            or "/.venv/" in path
            or "__pycache__" in path
            or path.endswith(".pem")
        )
    ]

    if dangerous:
        result(
            "FAIL",
            "Potentially sensitive/generated tracked paths: "
            + ", ".join(dangerous[:12]),
        )
    else:
        result(
            "PASS",
            "No .env, virtual environment, PEM keys "
            "or Python caches in tracked paths",
        )

    # Scan tracked, reasonably sized text files.
    # Never display any detected secret value.
    patterns = [
        re.compile(r"gsk_[A-Za-z0-9]{20,}"),
        re.compile(r"hf_[A-Za-z0-9]{25,}"),
    ]

    flagged = []

    for relative in tracked:
        path = ROOT / relative

        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue

        try:
            content = path.read_text(
                encoding="utf-8"
            )
        except (UnicodeDecodeError, OSError):
            continue

        if any(pattern.search(content) for pattern in patterns):
            flagged.append(relative)

    if flagged:
        result(
            "FAIL",
            "Possible API tokens in tracked files: "
            + ", ".join(flagged),
        )
    else:
        result("PASS", "No obvious API tokens found in tracked text")


def main():

    print("\n========== SUBMISSION PREFLIGHT ==========\n")

    check_files()
    check_csv()
    check_report()
    check_git()

    print("\n========== SUMMARY ==========")

    counts = Counter(level for level, _ in findings)

    print(dict(counts))

    if counts["FAIL"]:
        print("RESULT: BLOCKED — fix failed checks")
    elif counts["WARN"]:
        print("RESULT: NEEDS FINAL REVIEW")
    else:
        print("RESULT: AUTOMATED CHECKS PASSED")

    print(
        "\nThis does not prove factual correctness "
        "or compliance with the two-page limit."
    )


if __name__ == "__main__":
    main()
