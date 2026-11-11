
"""Create a two-page landscape A4 submission from saved results.

Run:
    python -m scripts.render_submission

No Groq calls. Does not modify results.csv.
"""

import csv
import html
import json

from collections import Counter
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "SUBMISSION_FINAL_2PAGES.pdf"

QUALITY_RUN = "quality_full_cca5739647"

RUN_IDS = [
    ("Steady", "steady_7b384c6aaa", 8),
    ("Burst", "burst_48688426ec", 24),
    ("Overload", "sustained_overload_44eea44af7", 48),
    ("RPM halved", "rpm_halved_e0cf5689a4", 24),
    ("Day exhausted", "daily_exhausted_32e13f1639", 12),
]

BLUE = colors.HexColor("#17324d")
MUTED = colors.HexColor("#475569")
LINE = colors.HexColor("#d5dfe8")
PALE = colors.HexColor("#eef3f8")

W, H = landscape(A4)
M = 27
FULL = W - 2 * M
GAP = 17
HALF = (FULL - GAP) / 2

STYLES = {
    "body": ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=8.6,
        leading=11.4,
        textColor=BLUE,
    ),
    "head": ParagraphStyle(
        "head",
        fontName="Helvetica-Bold",
        fontSize=9.8,
        leading=12.4,
        textColor=BLUE,
    ),
    "tiny": ParagraphStyle(
        "tiny",
        fontName="Helvetica",
        fontSize=7.9,
        leading=10.1,
        textColor=BLUE,
    ),
    "cell": ParagraphStyle(
        "cell",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.0,
        textColor=BLUE,
    ),
    "cellhead": ParagraphStyle(
        "cellhead",
        fontName="Helvetica-Bold",
        fontSize=7.3,
        leading=8.8,
        textColor=BLUE,
    ),
}


def safe(value):
    return html.escape(str(value), quote=False)


def paragraph(pdf, value, x, top, width, style="body"):
    item = Paragraph(safe(value), STYLES[style])
    _, height = item.wrap(width, 1000)
    item.drawOn(pdf, x, top - height)
    return top - height


def section(pdf, x, top, width, heading, body, style="body"):
    top = paragraph(
        pdf, heading, x, top, width, "head"
    ) - 3

    return paragraph(
        pdf, body, x, top, width, style
    )


def draw_table(pdf, top, headers, data, widths, x=M):
    content = [
        [
            Paragraph(safe(value), STYLES["cellhead"])
            for value in headers
        ]
    ]

    for row in data:
        content.append([
            Paragraph(safe(value), STYLES["cell"])
            for value in row
        ])

    item = Table(
        content,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.55, LINE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.32, LINE),
    ]))

    _, height = item.wrap(sum(widths), 1000)
    item.drawOn(pdf, x, top - height)

    return top - height


def caption(pdf, top, value):
    pdf.setFont("Helvetica-Bold", 7.6)
    pdf.setFillColor(MUTED)
    pdf.drawString(M, top - 8, value)

    return top - 14


def number(row, key):
    try:
        value = row.get(key)

        if value in (None, ""):
            return None

        return float(value)

    except (ValueError, TypeError):
        return None


def fmt(value, digits=1):
    if value is None:
        return "not measured"

    return f"{value:.{digits}f}"


def percentile_text(rows, key, points=(50, 95, 99)):
    values = sorted(
        value
        for row in rows
        if (value := number(row, key)) is not None
    )

    if not values:
        return "not measured"

    output = []

    for point in points:
        index = (len(values) - 1) * point / 100

        lower = int(index)
        upper = min(lower + 1, len(values) - 1)

        fraction = index - lower

        value = (
            values[lower] * (1 - fraction)
            + values[upper] * fraction
        )

        output.append(fmt(value, 0))

    return " / ".join(output)


def arrival_rate(rows):
    times = sorted(
        datetime.fromisoformat(row["start_time"])
        for row in rows
    )

    elapsed = (
        times[-1] - times[0]
    ).total_seconds()

    if elapsed <= 0:
        return None

    return 60 * (len(times) - 1) / elapsed


def duration(rows):
    candidates = [
        number(row, "duration_s")
        for row in rows
    ]

    candidates = [
        value for value in candidates
        if value is not None
    ]

    if candidates:
        return candidates[0]

    run_id = rows[0].get("run_id")

    path = (
        RESULTS / f"load_summary_{run_id}.json"
    )

    if path.exists():
        summary = json.loads(
            path.read_text(encoding="utf-8")
        )

        for key in (
            "duration_s",
            "duration",
            "elapsed_s",
            "elapsed_seconds",
        ):
            value = summary.get(key)

            if isinstance(value, (int, float)):
                return value

    return None


def load_table_rows(rows, name):
    successful = [
        row for row in rows
        if row["status"] == "ok"
    ]

    statuses = Counter(
        row["status"] for row in rows
    )

    tokens = [
        number(row, "tokens_in")
        + number(row, "tokens_out")
        for row in successful
        if (
            number(row, "tokens_in") is not None
            and number(row, "tokens_out") is not None
        )
    ]

    retry_values = [
        number(row, "retries")
        for row in rows
    ]

    mean_retries = (
        sum(retry_values) / len(retry_values)
        if all(v is not None for v in retry_values)
        else None
    )

    peaks = [
        number(row, "peak_queue")
        for row in rows
    ]

    peaks = [
        value for value in peaks
        if value is not None
    ]

    peak_queue = (
        int(max(peaks))
        if peaks else "not measured"
    )

    table_2a = [
        name,
        "stand-in",
        len(rows),
        fmt(duration(rows)),
        fmt(arrival_rate(rows)),
        percentile_text(rows=successful, key="first_token_ms"),
        percentile_text(rows=successful, key="total_ms"),
    ]

    def pct(value):
        return f"{100 * value / len(rows):.1f}%"

    table_2b = [
        name,
        pct(statuses["ok"]),
        pct(statuses["rate_limited"]),
        pct(statuses["timeout"] + statuses["error"]),
        fmt(mean_retries, 2),
        peak_queue,
        fmt(
            sum(tokens) / len(tokens)
            if tokens else None
        ),
        "not measured",
    ]

    return table_2a, table_2b


def page_header(pdf, page):
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 15)

    pdf.drawString(
        M, H - 28, "THE RELAY"
    )

    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(MUTED)

    pdf.drawRightString(
        W - M,
        H - 26,
        f"Hiring assignment | Page {page} of 2",
    )

    pdf.setStrokeColor(LINE)
    pdf.line(M, H - 34, W - M, H - 34)

    pdf.setFont("Helvetica", 7)

    pdf.drawString(
        M,
        18,
        "Real Groq measurements and labeled stand-in "
        "experiments | Raw data attached separately",
    )


def check_space(bottom, page):
    if bottom < 35:
        raise RuntimeError(
            f"Page {page} overflow at {bottom:.1f} pt"
        )


def main():
    with (RESULTS / "results.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = list(csv.DictReader(stream))

    by_run = {}

    for row in rows:
        by_run.setdefault(
            row.get("run_id"), []
        ).append(row)

    for name, run_id, count in RUN_IDS:
        if len(by_run.get(run_id, [])) != count:
            raise ValueError(
                f"{name}: expected {count} records"
            )

    quality = by_run.get(QUALITY_RUN, [])

    if len(quality) != 45:
        raise ValueError(
            "Expected 45 real Groq evaluation records"
        )

    audit = json.loads(
        (
            RESULTS / "FINAL_MEASUREMENT_AUDIT.json"
        ).read_text(encoding="utf-8")
    )

    slow = audit["slow_trace"]
    retry = audit["retry"]

    if not slow:
        raise ValueError("No recorded slow trace")

    retry_penalty = (
        retry["retry_naive"]["total_ms"]
        - retry["retry_no_retry"]["total_ms"]
    )

    table_2a = []
    table_2b = []

    for name, run_id, _ in RUN_IDS:
        a, b = load_table_rows(
            by_run[run_id], name
        )

        table_2a.append(a)
        table_2b.append(b)

    table_3 = []

    settings = [
        ("small", "250w / 2 / no"),
        ("balanced", "500w / 4 / no"),
        ("reranked", "500w / 4 / lexical"),
    ]

    for name, description in settings:
        group = [
            row for row in quality
            if row["retrieval_setting"] == name
        ]

        if len(group) != 15:
            raise ValueError(
                f"Incomplete retrieval setting: {name}"
            )

        if any(
            not row.get("answer_grade")
            for row in group
        ):
            raise ValueError(
                f"Missing grades for {name}"
            )

        grades = Counter(
            row["answer_grade"] for row in group
        )

        tokens = [
            number(row, "tokens_in")
            + number(row, "tokens_out")
            for row in group
            if (
                number(row, "tokens_in") is not None
                and number(row, "tokens_out") is not None
            )
        ]

        table_3.append([
            f"{name}: {description}",
            "15",
            f"{100 * grades['correct'] / 15:.1f}%",
            f"{100 * (grades['wrong'] + grades['refused_wrongly']) / 15:.1f}%",
            f"{100 * grades['refused_correctly'] / 15:.1f}%",
            fmt(
                sum(tokens) / len(tokens)
                if tokens else None
            ),
            percentile_text(
                group,
                "provider_first_content_ms",
                (50, 95),
            ),
            percentile_text(
                group,
                "total_ms",
                (50, 95),
            ),
        ])

    pdf = Canvas(
        str(OUTPUT),
        pagesize=(W, H),
        pageCompression=1,
    )

    pdf.setTitle(
        "The Relay - Two-page submission"
    )

    # ---------------- PAGE 1 ----------------

    page_header(pdf, 1)

    top = H - 49

    left = section(
        pdf, M, top, HALF,
        "1. What I built and why",
        "Python five-stage research QA: route, retrieve "
        "with BM25, build grounded prompts, stream Groq "
        "output, and validate citations. A citation-repair "
        "branch is available. I prioritize grounded answers "
        "over accepting every request; paper scope must "
        "constrain retrieval. Ignoring it caused three "
        "wrongful Transformer refusals in the baseline.",
    ) - 8

    left = section(
        pdf, M, left, HALF,
        "2. The real service and my stand-in",
        "Groq openai/gpt-oss-20b: 3 successful calibration "
        "calls gave model first-content median 884.57 ms "
        "and total median 983.85 ms; the sample cannot "
        "establish tail behavior. Earlier low-output-cap "
        "tests truncated. The FastAPI stand-in streams SSE "
        "and simulates quotas, concurrency, failures and "
        "slow tails from these 3 samples. Its quota/token "
        "model is approximate; lab TPM 50,000 differs "
        "from real 8,000.",
    )

    right_x = M + HALF + GAP

    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 8.9)

    pdf.drawString(
        right_x, top - 10,
        "Table 1. Real-service limits"
    )

    table_1_bottom = draw_table(
        pdf,
        top - 17,
        ["Limit", "Value", "Evidence", "Confidence"],
        [
            ["Requests/min", "30", "Groq documentation", "medium"],
            ["Tokens/min", "8,000", "Headers + docs", "high"],
            ["Requests/day", "1,000", "Headers + docs", "high"],
            ["Tokens/day", "200,000", "Groq docs only", "medium"],
            ["Max concurrency", "not measured", "No real saturation", "low"],
            ["Limit response", "429 (docs)", "Not observed live", "medium"],
            ["Recovery time", "not measured", "No live 429", "low"],
        ],
        [110, 70, 136, 67],
        x=right_x,
    )

    y = min(left, table_1_bottom) - 9

    y = section(
        pdf, M, y, FULL,
        "3. Results",
        "Five load shapes used stand-in only. Percentiles "
        "use successful requests. Controlled 30 vs 15 RPM "
        "(24 requests each): successes fell from 24 to 15, "
        "a 37.5 percentage-point drop. This isolates RPM "
        "better than the original confounded comparison.",
        "tiny",
    ) - 4

    y = caption(
        pdf, y,
        "Table 2a. Latency | p50 / p95 / p99 milliseconds"
    )

    y = draw_table(
        pdf, y,
        [
            "Scenario", "Source", "N", "Duration (s)",
            "Offered /min", "First model content (ms)",
            "Total latency (ms)",
        ],
        table_2a,
        [118, 55, 27, 72, 72, 219, 224],
    ) - 6

    y = caption(
        pdf, y,
        "Table 2b. Outcomes | timeout/gave-up includes timeout and error"
    )

    y = draw_table(
        pdf, y,
        [
            "Scenario", "OK", "Rate limited",
            "Timeout / gave up", "Retries / request",
            "Peak queue", "Tokens / OK answer",
            "Cost / 1k",
        ],
        table_2b,
        [118, 54, 50, 113, 95, 76, 145, 76],
    )

    check_space(y, 1)
    pdf.showPage()

    # ---------------- PAGE 2 ----------------

    page_header(pdf, 2)

    top = H - 49

    left = section(
        pdf, M, top, HALF,
        "4. Where the time goes",
        f"Slow stand-in request {slow['request_id']}: "
        f"queue {slow['queue_ms']:.2f} ms; "
        f"pre-first-content work "
        f"{slow['working_before_first_model_content_ms']:.2f} ms; "
        f"post-first-content work "
        f"{slow['working_after_first_model_content_ms']:.2f} ms; "
        f"retries {slow['retries']}; "
        f"total {slow['total_ms']:.2f} ms. "
        f"Model content began at "
        f"{slow['first_model_content_ms']:.2f} ms; "
        f"buffered reply ready at "
        f"{slow['buffered_ready_ms']:.2f} ms. "
        "Work includes HTTP/application overhead.",
    )

    right = section(
        pdf, right_x, top, HALF,
        "5. When a limit is hit",
        "Bounded FIFO admission reserves request/token "
        "capacity. At a minute limit, eligible callers "
        "queue until their deadline; excess/expired "
        "requests are rejected. At the daily limit, "
        "immediate rejection is preferable to waiting "
        "or a weaker, ungrounded answer. Recovery time "
        "is unknown on the real service.",
    )

    y = min(left, right) - 10

    y = section(
        pdf, M, y, FULL,
        "Retrieval: quality versus cost",
        "The same 15 questions were run under three settings "
        "on real Groq (45 calls): 6 existing human decisions, "
        "39 offline rubric decisions. The rubric checks "
        "core concepts and citation provenance, not every "
        "claim. Balanced is preferred for its less extreme "
        "observed latency tail; small saves tokens near "
        "TPM pressure (manual choice, not automatic).",
        "tiny",
    ) - 4

    y = caption(
        pdf, y,
        "Table 3. Real Groq retrieval quality | rubric-screened grades; provider first content"
    )

    y = draw_table(
        pdf, y,
        [
            "Setting: words/top-k/rerank",
            "N",
            "Correct + cited*",
            "Wrong incl. false refusal",
            "Proper refusal",
            "Tokens / answer",
            "First content p50/p95 (ms)",
            "Total p50/p95 (ms)",
        ],
        table_3,
        [146, 25, 94, 108, 80, 89, 119, 116],
    ) - 9

    left = section(
        pdf, M, y, HALF,
        "How this library will mislead a naive retriever",
        "(1) Shared architecture terms: the Transformer "
        "encoder/decoder question retrieved H-Net and "
        "caused three refusals. (2) H-Net figures contain "
        "garbled extracted glyphs, so chunk-boundary "
        "queries rank noisy evidence. (3) An unanswerable "
        "stock-price query retrieved Moshi through "
        "incidental word overlap.",
        "tiny",
    ) - 7

    left = section(
        pdf, M, left, HALF,
        "6. Retries",
        "For transient failures, respect Retry-After and "
        "the caller's deadline. Never retry daily quota "
        "exhaustion or a partial streamed response. "
        f"In a simulated 1-RPM test, naive retries added "
        f"{retry_penalty:.2f} ms and two attempts with "
        "no recovered answer. The retry module was tested "
        "separately, not wired into normal requests.",
        "tiny",
    ) - 7

    left = section(
        pdf, M, left, HALF,
        "7. What the person sees",
        "At 3pm with no daily budget: 'Today's research-answer "
        "budget is exhausted. I cannot verify an answer "
        "now and will not guess. Please try after the "
        "quota resets.' Proposed wording, not a tested UI.",
        "tiny",
    )

    right = section(
        pdf, right_x, y, HALF,
        "8. What I'm unsure about, what I left out, what I learned",
        "Three successful real latency samples cannot "
        "characterize burst tails; the stand-in depends "
        "on that assumption. I deliberately omitted "
        "a vector database and production UI. I learned "
        "that retry budgets must account for user patience, "
        "not only HTTP statuses.",
        "tiny",
    ) - 7

    right = section(
        pdf, right_x, right, HALF,
        "Your feedback on Vaani",
        "Website-only impression (verify personally): "
        "multilingual voice and turn-taking stood out; "
        "different latency claims were confusing. "
        "I would define each timer's start and stop "
        "beside the claim. I did not test the product.",
        "tiny",
    ) - 7

    right = section(
        pdf, right_x, right, HALF,
        "How I worked with the assistant",
        "Three prompts mattered: ground answers in exactly "
        "three papers (avoid guessing); provide complete "
        "copy-paste scripts/tests (reproducibility); assess "
        "answers rather than require 45 manual grades "
        "(reduce unnecessary work). I identified citation "
        "parsing failures from traces and overruled "
        "manual per-answer grading. I ran and diagnosed "
        "the experiments; the code was AI-assisted. "
        "Attach the unedited chat history separately.",
        "tiny",
    )

    check_space(min(left, right), 2)

    pdf.save()

    # Automated PDF validation.
    import fitz

    document = fitz.open(OUTPUT)

    if len(document) != 2:
        raise RuntimeError(
            f"Expected 2 pages, found {len(document)}"
        )

    content = "\n".join(
        page.get_text() for page in document
    )

    required_headings = [
        "What I built and why",
        "The real service and my stand-in",
        "Results",
        "Where the time goes",
        "When a limit is hit",
        "Retrieval: quality versus cost",
        "How this library will mislead a naive retriever",
        "Retries",
        "What the person sees",
        "What I'm unsure about",
        "Your feedback on Vaani",
        "How I worked with the assistant",
    ]

    missing = [
        heading for heading in required_headings
        if heading not in content
    ]

    document.close()

    if missing:
        raise RuntimeError(
            f"PDF missing headings: {missing}"
        )

    print("\n========== FINAL SUBMISSION PDF ==========")
    print("Created:", OUTPUT)
    print("Pages: 2")
    print("Required headings: 12/12")
    print("Table 1: included")
    print("Table 2a: included")
    print("Table 2b: included")
    print("Table 3: included")
    print("Original results.csv: unchanged")
    print("\nInspect the PDF visually before submission.")


if __name__ == "__main__":
    main()
