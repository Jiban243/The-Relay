
"""The Relay: real-model retrieval and quality evaluation.

Commands:
    python -m evaluation.run_quality --mode inspect
    python -m evaluation.run_quality --mode pilot
    python -m evaluation.run_quality --mode full

The inspect mode uses no Groq API calls.
The pilot mode runs 3 real requests.
The full mode evaluates 15 questions x 3 settings.

Final answer grading is manual, not automatic.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from experiments.run_load import FIELDS, append_csv
from src.pipeline import RelayPipeline


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

SETTINGS = ["small", "balanced", "reranked"]


# Initial question candidates.
# Review these against the actual papers before final use.
QUESTIONS = [
    {
        "id": "a1",
        "paper": "1706.03762v7",
        "supported": True,
        "question": (
            "According to Attention Is All You Need, "
            "how is scaled dot-product attention computed?"
        ),
    },
    {
        "id": "a2",
        "paper": "1706.03762v7",
        "supported": True,
        "question": (
            "What are the encoder and decoder components "
            "of the Transformer?"
        ),
    },
    {
        "id": "a3",
        "paper": "1706.03762v7",
        "supported": True,
        "question": (
            "Why does Attention Is All You Need "
            "use positional encodings?"
        ),
    },
    {
        "id": "a4",
        "paper": "1706.03762v7",
        "supported": True,
        "question": (
            "How does multi-head attention work "
            "in Attention Is All You Need?"
        ),
    },
    {
        "id": "m1",
        "paper": "2410.00037v1",
        "supported": True,
        "question": (
            "How does Moshi support full-duplex "
            "spoken dialogue?"
        ),
    },
    {
        "id": "m2",
        "paper": "2410.00037v1",
        "supported": True,
        "question": (
            "What is the role of Mimi "
            "in the Moshi architecture?"
        ),
    },
    {
        "id": "m3",
        "paper": "2410.00037v1",
        "supported": True,
        "question": (
            "How does Moshi use semantic "
            "and acoustic tokens?"
        ),
    },
    {
        "id": "m4",
        "paper": "2410.00037v1",
        "supported": True,
        "question": (
            "What is the inner-monologue mechanism "
            "described in Moshi?"
        ),
    },
    {
        "id": "h1",
        "paper": "2507.07955v2",
        "supported": True,
        "question": (
            "How does H-Net perform dynamic chunking?"
        ),
    },
    {
        "id": "h2",
        "paper": "2507.07955v2",
        "supported": True,
        "question": (
            "How does H-Net learn where "
            "to place chunk boundaries?"
        ),
    },
    {
        "id": "h3",
        "paper": "2507.07955v2",
        "supported": True,
        "question": (
            "What is the purpose of the routing "
            "module in H-Net?"
        ),
    },
    {
        "id": "h4",
        "paper": "2507.07955v2",
        "supported": True,
        "question": (
            "What roles do the encoder, main network "
            "and decoder play in H-Net?"
        ),
    },
    {
        "id": "u1",
        "paper": None,
        "supported": False,
        "question": (
            "What was Apple's closing stock price "
            "on November 10, 2026?"
        ),
    },
    {
        "id": "u2",
        "paper": None,
        "supported": False,
        "question": (
            "What was the weather in London "
            "on November 10, 2026?"
        ),
    },
    {
        "id": "u3",
        "paper": None,
        "supported": False,
        "question": (
            "What was Groq's total annual revenue "
            "in 2026?"
        ),
    },
]


def validate_questions():
    ids = [q["id"] for q in QUESTIONS]

    assert len(QUESTIONS) >= 15
    assert len(ids) == len(set(ids))

    assert sum(q["supported"] for q in QUESTIONS) >= 12
    assert sum(not q["supported"] for q in QUESTIONS) >= 3


def select_jobs(mode):
    if mode == "pilot":
        pilot_ids = {"a1", "m1", "h1"}

        return [
            (q, "balanced")
            for q in QUESTIONS
            if q["id"] in pilot_ids
        ]

    if mode == "full":
        jobs = []

        # Rotate setting order to reduce order effects.
        for index, question in enumerate(QUESTIONS):
            for shift in range(len(SETTINGS)):
                setting = SETTINGS[
                    (index + shift) % len(SETTINGS)
                ]

                jobs.append((question, setting))

        return jobs

    return []


def make_csv_row(run_id, question, setting, trace):
    row = {field: "" for field in FIELDS}

    model_calls = trace.get("model_calls", [])

    final_error = ""
    if model_calls:
        final_error = (
            model_calls[-1].get("error") or ""
        )

    # A locally produced refusal uses zero model tokens.
    # Missing usage from an attempted call stays blank.
    tokens_in = trace.get("tokens_in")
    tokens_out = trace.get("tokens_out")

    if not model_calls:
        tokens_in = 0
        tokens_out = 0

    row.update({
        "request_id": trace["request_id"],
        "scenario": "retrieval_quality",
        "source": "real",
        "start_time": trace["start_time"],
        "first_token_ms": trace["first_token_ms"],
        "total_ms": trace["total_ms"],
        "status": trace["status"],
        "retries": trace["retries"],
        "tokens_in": (
            "" if tokens_in is None else tokens_in
        ),
        "tokens_out": (
            "" if tokens_out is None else tokens_out
        ),
        "retrieval_setting": setting,

        # Must be filled through human review.
        "answer_grade": "",

        "run_id": run_id,
        "question_id": question["id"],
        "admission_reason": "not_applicable",
        "queue_wait_ms": trace.get(
            "queue_wait_ms", 0
        ),
        "provider_first_content_ms": (
            trace.get("provider_first_content_ms") or ""
        ),
        "user_visible_first_content_ms": (
            trace["first_token_ms"]
        ),
        "retrieval_ms": trace["stages_ms"].get(
            "retrieval", ""
        ),
        "model_ms": trace["stages_ms"].get(
            "model_calls", ""
        ),
        "revisions": trace.get("revisions", 0),
        "error_detail": final_error,
    })

    return row


def inspect_retrieval(pipeline):
    print("\n========== EVIDENCE INSPECTION ==========")
    print("No Groq calls will be made.\n")

    for question in QUESTIONS:
        print(f"\n[{question['id']}] {question['question']}")
        print(
            "Expected paper:",
            question["paper"] or "None"
        )
        print("Expected supported:", question["supported"])

        passages = pipeline.retriever.search(
            question["question"],
            setting="balanced",
        )

        if not passages:
            print("No passages retrieved.")
            continue

        for passage in passages[:2]:
            print(
                f"  {passage.citation} "
                f"score={passage.score:.2f}"
            )
            print(
                " ",
                " ".join(passage.text.split())[:160],
            )


async def execute(mode, pace_s, run_id=None, resume=False):
    validate_questions()

    if mode == "inspect":
        pipeline = RelayPipeline(source="stand-in")

        try:
            inspect_retrieval(pipeline)
        finally:
            await pipeline.close()

        return

    jobs = select_jobs(mode)

    if run_id is None:
        run_id = (
            f"quality_{mode}_{uuid4().hex[:10]}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RESULTS_DIR / f"{run_id}.jsonl"

    completed = set()

    if output_path.exists():
        if not resume:
            raise RuntimeError(
                f"Existing run file: {output_path}. "
                "Use --resume with the same --run-id "
                "to continue without repeating completed jobs."
            )

        with output_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                completed.add((
                    record["question_id"],
                    record["retrieval_setting"],
                ))

    pipeline = RelayPipeline(source="real")
    started = perf_counter()
    executed = 0

    print("\n========== QUALITY EVALUATION ==========")
    print("Mode:", mode)
    print("Run ID:", run_id)
    print("Model:", pipeline.model)
    print("Evaluation jobs:", len(jobs))
    print("Previously completed:", len(completed))
    print("Minimum spacing:", pace_s, "seconds")
    print("Source: REAL GROQ")
    print("No automatic provider retries or revisions.\n")

    try:
        for index, (question, setting) in enumerate(
            jobs, start=1
        ):
            key = (question["id"], setting)

            if key in completed:
                continue

            call_started = perf_counter()

            # Store the exact passages used for human review.
            passages = pipeline.retriever.search(
                question["question"],
                setting=setting,
            )

            evidence = [
                {
                    "chunk_id": passage.chunk_id,
                    "citation": passage.citation,
                    "score": passage.score,
                    "text": passage.text,
                }
                for passage in passages
            ]

            trace = await pipeline.answer(
                question=question["question"],
                setting=setting,
                scenario="retrieval_quality",
                allow_revision=False,
            )

            record = {
                "run_id": run_id,
                "question_id": question["id"],
                "question": question["question"],
                "expected_paper": question["paper"],
                "expected_supported": question["supported"],
                "retrieval_setting": setting,
                "evidence": evidence,
                "trace": trace,
                "human_grade": None,
                "human_notes": "",
            }

            # Persist each result before the next request.
            with output_path.open(
                "a", encoding="utf-8"
            ) as file:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

            append_csv([
                make_csv_row(
                    run_id,
                    question,
                    setting,
                    trace,
                )
            ])

            executed += 1
            completed.add(key)

            print(
                f"[{index}/{len(jobs)}] "
                f"{question['id']} | {setting}"
            )

            print(
                "Status:",
                trace["status"],
                "| Verification:",
                trace["verification"],
            )

            print(
                "Latency:",
                trace["total_ms"],
                "ms | Tokens:",
                trace["tokens_in"],
                "+",
                trace["tokens_out"],
            )

            print(
                "Citations:",
                ", ".join(
                    trace["retrieved_citations"]
                ),
            )

            print(
                "Answer:",
                " ".join(
                    trace["answer"].split()
                )[:280],
            )

            print("-" * 55)

            # Stop rather than wasting quota on
            # repeated errors or retry storms.
            if trace["status"] != "ok":
                print(
                    "\nStopping after model failure. "
                    "The raw result has been saved."
                )
                break

            # Conservative quota protection.
            model_calls = trace.get("model_calls") or []

            if model_calls:
                headers = model_calls[-1].get(
                    "rate_headers"
                ) or {}

                remaining = headers.get(
                    "x-ratelimit-remaining-tokens"
                )

                if remaining is not None:
                    try:
                        if int(remaining) < 1200:
                            print(
                                "Low TPM remaining. "
                                "Allowing extra recovery time."
                            )
                            await asyncio.sleep(60)
                    except ValueError:
                        pass

            # Maintain spacing between real requests.
            elapsed = perf_counter() - call_started

            if index < len(jobs):
                await asyncio.sleep(
                    max(0.0, pace_s - elapsed)
                )

    finally:
        await pipeline.close()

    print("\n========== RUN SUMMARY ==========")
    print("Run ID:", run_id)
    print("New evaluations:", executed)
    print(
        "Elapsed:",
        round(perf_counter() - started, 2),
        "seconds",
    )
    print("Raw evidence:", output_path)
    print("Request measurements:", RESULTS_DIR / "results.csv")
    print("Answer grades: awaiting manual review")

    if len(completed) < len(jobs):
        print(
            "\nTo resume this run:"
            f"\npython -m evaluation.run_quality "
            f"--mode {mode} --run-id {run_id} --resume"
        )


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["inspect", "pilot", "full"],
        default="inspect",
    )

    parser.add_argument(
        "--pace",
        type=float,
        default=30.0,
        help="Minimum seconds between real API requests",
    )

    parser.add_argument(
        "--run-id",
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.pace < 0:
        parser.error("--pace must be nonnegative")

    await execute(
        mode=args.mode,
        pace_s=args.pace,
        run_id=args.run_id,
        resume=args.resume,
    )


if __name__ == "__main__":
    asyncio.run(main())
