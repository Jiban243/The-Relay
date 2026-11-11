
"""Run one instrumented request through The Relay."""

import argparse
import asyncio
import json

from pathlib import Path

from src.pipeline import RelayPipeline


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT / "results" / "pipeline_requests.jsonl"
)


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=["stand-in", "real"],
        default="stand-in",
    )

    parser.add_argument(
        "--setting",
        choices=["small", "balanced", "reranked"],
        default="balanced",
    )

    parser.add_argument(
        "--question",
        default=(
            "According to Moshi, "
            "how does full-duplex dialogue work?"
        ),
    )

    parser.add_argument(
        "--no-revision",
        action="store_true",
    )

    args = parser.parse_args()

    pipeline = RelayPipeline(
        source=args.source
    )

    try:
        record = await pipeline.answer(
            question=args.question,
            setting=args.setting,
            scenario="manual_pipeline_smoke_test",
            allow_revision=not args.no_revision,
        )

    finally:
        await pipeline.close()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

    print("\n========== THE RELAY ==========")

    print("Source:", record["source"])
    print("Status:", record["status"])
    print("Route:", record["route"])

    print(
        "Retrieved:",
        ", ".join(
            record["retrieved_citations"]
        )
    )

    print(
        "Verification:",
        record["verification"]
    )

    print(
        "Model calls:",
        len(record["model_calls"])
    )

    print("Retries:", record["retries"])

    print(
        "Provider first content (ms):",
        record["provider_first_content_ms"]
    )

    print(
        "User-visible first content (ms):",
        record["first_token_ms"]
    )

    print(
        "Total time (ms):",
        record["total_ms"]
    )

    print(
        "Tokens in/out:",
        record["tokens_in"],
        record["tokens_out"]
    )

    print(
        "Stages (ms):",
        record["stages_ms"]
    )

    print("\nAnswer:\n")
    print(record["answer"])

    print("\nSaved trace:", OUTPUT)

    if args.source == "stand-in":

        print(
            "\nNOTE: Simulated output is not a "
            "factual answer and must not be "
            "quality-graded."
        )


if __name__ == "__main__":
    asyncio.run(main())
