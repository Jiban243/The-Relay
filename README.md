# The Relay

A paper-grounded, multi-stage research assistant built for the Vaani AI hiring assignment.

## Research library

The assistant is restricted to these exact arXiv versions:

- Attention Is All You Need — 1706.03762v7
- Moshi — 2410.00037v1
- Dynamic Chunking / H-Net — 2507.07955v2

The PDFs, source manifest and extracted page corpus are stored under `data/`.

## Architecture

The Python pipeline routes a question, retrieves BM25 evidence, constructs a grounded prompt, calls a streaming model, and checks citation provenance.

Paper-scoped retrieval prevents unrelated documents from dominating queries that explicitly identify a paper. A citation-repair branch is available.

Citation validation checks provenance, not full factual correctness.

## Setup

Create a Python virtual environment and install dependencies:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

For real Groq calls, create a local `.env` containing:

    GROQ_API_KEY=your_key_here
    GROQ_MODEL=openai/gpt-oss-20b

Never commit the real `.env` file.

## Reproduce checks

    python -m pytest tests/ -q
    python -m scripts.verify_papers
    python -m scripts.check_paper_scope

## Reproduce stand-in experiments

    python -m experiments.run_load --scenario all
    python -m experiments.compare_rpm
    python -m experiments.retry_experiment

The stand-in is a controlled approximation and does not consume Groq quota.

These commands generate new run IDs. Historical measurements used in the submission remain in `results/`.

## Reproduce the saved-results audit

    python -m scripts.audit_results
    python -m scripts.build_submission
    python -m scripts.render_submission
    python -m scripts.submission_preflight

The saved-results audit does not call any model API.

## Real-model quality evaluation

The historical benchmark contains 15 questions tested with each of three retrieval settings, producing 45 real Groq responses.

Run metadata and full retrieved evidence are saved in:

    results/quality_full_cca5739647.jsonl

The corresponding request records are in:

    results/results.csv

Six grades were recorded during human review. The remaining 39 were produced by a transparent offline answer-concept and citation rubric. These scores are not independent expert verification of every factual claim.

Do not rerun the full real-model evaluation unless necessary; it consumes real Groq quota.

## Main submission files

- `SUBMISSION_FINAL_2PAGES.pdf` — required two-page write-up
- `results/results.csv` — request-level measurements
- `results/FINAL_MEASUREMENT_AUDIT.md` — experiment tables
- `results/QUALITY_GRADE_AUDIT.jsonl` — grading provenance
- `results/quality_full_cca5739647.jsonl` — full real-model answers
- `src/`, `scripts/`, `experiments/`, `evaluation/`, `tests/` — implementation and tests

## Measurement limitations

The stand-in was calibrated using three successful Groq latency samples, insufficient to characterize production tail behavior. Its limits and injected failures are simulated.

The pipeline buffers answers for citation validation. Model first-content latency must not be confused with user-visible response readiness.

The original pre-fix evaluation is retained unchanged. Subsequent routing fixes are regression-tested separately.

## Security

API keys and virtual environments are excluded through `.gitignore`.
