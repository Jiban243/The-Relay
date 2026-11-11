
"""The Relay: grounded pipeline with paper-scoped retrieval."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from src.evidence import REFUSAL, make_grounded_messages
from src.query_router import route_question
from src.retriever import BM25Retriever, SETTINGS


ROOT = Path(__file__).resolve().parents[1]

TRANSFORMER_ID = "1706.03762v7"

CITE_PATTERN = re.compile(
    r"(?P<open>[\[【])\s*"
    r"(?P<paper>\d{4}\.\d{4,5}v\d+)\s*,\s*"
    r"PDF\s*p\.\s*(?P<page>\d+)\s*"
    r"(?P<close>[\]】])",
    re.IGNORECASE,
)

CITATION_LIKE_PATTERN = re.compile(
    r"[\[【][^\]】]{0,160}"
    r"(?:PDF\s*p\.|\d{4}\.\d{4,5}v\d+)"
    r"[^\]】]{0,160}[\]】]",
    re.IGNORECASE,
)

BRACKET_PAIRS = {
    "[": "]",
    "【": "】",
}


def select_paper_scope(question, route=None):
    """
    Decide which paper, if any, retrieval should use.

    Explicit paper titles and arXiv IDs are handled
    by the existing question router.

    The additional Transformer alias fixes the
    observed a2 retrieval failure.

    Ambiguous questions search all three papers.
    """

    if route is None:
        route = route_question(question)

    explicit = getattr(route, "paper_id", None)

    if explicit:
        return explicit, "explicit_paper"

    q = question.casefold()

    if (
        re.search(
            r"\b(?:the|original)\s+transformer\b",
            q,
        )
        and not re.search(
            r"\b(?:moshi|h[ -]?net)\b",
            q,
        )
    ):
        return TRANSFORMER_ID, "transformer_alias"

    return None, "all_three_papers"


def extract_citations(answer):
    """Return cited paper ID and page pairs."""

    return [
        (
            match.group("paper"),
            int(match.group("page")),
        )
        for match in CITE_PATTERN.finditer(answer)
    ]


def check_citations(answer, passages):
    """
    Validate citation provenance.

    This verifies that referenced paper pages were
    retrieved; it does not prove factual correctness.
    """

    matches = list(CITE_PATTERN.finditer(answer))

    if not matches:
        return False

    citation_like = (
        CITATION_LIKE_PATTERN.findall(answer)
    )

    if len(matches) != len(citation_like):
        return False

    permitted = {
        (
            passage.paper_id,
            int(passage.pdf_page),
        )
        for passage in passages
    }

    for match in matches:

        opening = match.group("open")
        closing = match.group("close")

        if BRACKET_PAIRS[opening] != closing:
            return False

        citation = (
            match.group("paper"),
            int(match.group("page")),
        )

        if citation not in permitted:
            return False

    return True


def normalize_citations(answer):
    """Normalize accepted citations to square brackets."""

    def replace_match(match):
        return (
            f"[{match.group('paper')}, "
            f"PDF p. {int(match.group('page'))}]"
        )

    return CITE_PATTERN.sub(replace_match, answer)


def looks_like_refusal(answer):
    """Recognize evidence-insufficiency refusals."""

    text = answer.strip().casefold()

    return text.startswith((
        "i cannot verify",
        "i can't verify",
        "the provided passages do not",
        "the supplied passages do not",
        "the provided papers do not",
        "the supplied papers do not",
        "the papers do not",
        "the excerpts do not",
        "there is insufficient evidence",
        "insufficient evidence",
        "this information is not present",
    ))


class RelayPipeline:

    def __init__(self, source="stand-in"):

        if source not in ("real", "stand-in"):
            raise ValueError(
                "source must be 'real' or 'stand-in'"
            )

        load_dotenv(ROOT / ".env")

        self.source = source

        if source == "real":

            self.model = os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            )

            self.url = (
                "https://api.groq.com/openai/v1/"
                "chat/completions"
            )

            self.api_key = os.getenv("GROQ_API_KEY")

            if not self.api_key:
                raise ValueError(
                    "Missing GROQ_API_KEY in .env"
                )

        else:

            self.model = "stand-in"

            self.url = (
                "http://127.0.0.1:8000/"
                "openai/v1/chat/completions"
            )

            self.api_key = None

        self.retriever = BM25Retriever(
            ROOT / "data/processed/pages.jsonl"
        )

        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                45.0,
                connect=10.0,
            ),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=25,
            ),
        )

    async def close(self):
        await self.http.aclose()

    async def _model_call(
        self,
        messages,
        max_tokens=768,
    ):
        """
        Make one streaming model request.

        No automatic provider retries occur here.
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": 0.1,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }

        if self.source == "real":

            payload.update({
                "reasoning_effort": "low",
                "include_reasoning": False,
            })

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        started = perf_counter()

        result = {
            "status": "error",
            "text": "",
            "first_content_ms": None,
            "total_ms": None,
            "tokens_in": None,
            "tokens_out": None,
            "http_status": None,
            "finish_reason": None,
            "rate_headers": {},
            "error": None,
        }

        pieces = []

        try:

            async with self.http.stream(
                "POST",
                self.url,
                headers=headers,
                json=payload,
            ) as response:

                result["http_status"] = (
                    response.status_code
                )

                result["rate_headers"] = {
                    k.lower(): v
                    for k, v in response.headers.items()
                    if (
                        k.lower().startswith("x-ratelimit")
                        or k.lower() == "retry-after"
                    )
                }

                if response.status_code >= 400:

                    body = (
                        await response.aread()
                    ).decode(errors="replace")

                    if response.status_code == 429:
                        result["status"] = "rate_limited"

                    elif response.status_code == 504:
                        result["status"] = "timeout"

                    else:
                        result["status"] = "error"

                    result["error"] = body[:400]

                else:

                    async for line in response.aiter_lines():

                        if not line.startswith("data:"):
                            continue

                        data = line[5:].strip()

                        if data == "[DONE]":
                            break

                        try:
                            event = json.loads(data)

                        except json.JSONDecodeError:
                            continue

                        if event.get("error"):
                            result["error"] = str(
                                event["error"]
                            )[:400]
                            continue

                        choices = event.get(
                            "choices"
                        ) or []

                        for choice in choices:

                            delta = choice.get(
                                "delta"
                            ) or {}

                            piece = (
                                delta.get("content") or ""
                            )

                            if piece:

                                if (
                                    result["first_content_ms"]
                                    is None
                                ):
                                    result["first_content_ms"] = (
                                        round(
                                            (
                                                perf_counter()
                                                - started
                                            ) * 1000,
                                            2,
                                        )
                                    )

                                pieces.append(piece)

                            if choice.get("finish_reason"):
                                result["finish_reason"] = (
                                    choice["finish_reason"]
                                )

                        usage = event.get("usage")

                        if not usage:
                            usage = (
                                event.get("x_groq") or {}
                            ).get("usage")

                        if usage:

                            result["tokens_in"] = (
                                usage.get("prompt_tokens")
                            )

                            result["tokens_out"] = (
                                usage.get(
                                    "completion_tokens"
                                )
                            )

                    result["text"] = "".join(pieces)

                    if result["error"]:

                        result["status"] = "error"

                    elif (
                        result["text"].strip()
                        and result["finish_reason"] == "stop"
                    ):
                        result["status"] = "ok"

                    else:
                        result["status"] = "error"
                        result["error"] = (
                            "Empty or truncated completion"
                        )

        except httpx.TimeoutException as exc:

            result["status"] = "timeout"
            result["error"] = type(exc).__name__

        except httpx.HTTPError as exc:

            result["status"] = "error"
            result["error"] = type(exc).__name__

        finally:

            result["total_ms"] = round(
                (perf_counter() - started) * 1000,
                2,
            )

        return result

    async def answer(
        self,
        question,
        setting="balanced",
        scenario="manual",
        allow_revision=True,
    ):

        if setting not in SETTINGS:
            raise ValueError(
                f"Unknown retrieval setting: {setting}"
            )

        if not question.strip():
            raise ValueError(
                "Question cannot be empty"
            )

        started = perf_counter()

        record = {
            "request_id": str(uuid4()),
            "scenario": scenario,
            "source": self.source,
            "start_time": datetime.now(
                timezone.utc
            ).isoformat(),
            "first_token_ms": None,
            "total_ms": None,
            "status": "error",
            "retries": 0,
            "revisions": 0,
            "tokens_in": None,
            "tokens_out": None,
            "retrieval_setting": setting,
            "answer_grade": None,
            "question": question,
            "answer": "",
            "route": None,
            "paper_scope": None,
            "scope_reason": None,
            "retrieved_citations": [],
            "recognized_citations": [],
            "verification": "not_checked",
            "provider_first_content_ms": None,
            "queue_wait_ms": 0.0,
            "stages_ms": {},
            "model_calls": [],
        }

        def finish(
            answer_text,
            status,
            verification,
        ):

            record["answer"] = answer_text
            record["status"] = status
            record["verification"] = verification

            record["total_ms"] = round(
                (perf_counter() - started) * 1000,
                2,
            )

            # Answer is buffered until validation.
            # This is response-ready latency, not a
            # separately observed network TTFT.
            record["first_token_ms"] = (
                record["total_ms"]
            )

            return record

        # ------------------------------------
        # STEP 1: ROUTE QUESTION
        # ------------------------------------

        step = perf_counter()

        route = route_question(question)

        paper_id, reason = select_paper_scope(
            question,
            route,
        )

        record["route"] = route.decision
        record["paper_scope"] = paper_id
        record["scope_reason"] = reason

        record["stages_ms"]["routing"] = round(
            (perf_counter() - step) * 1000,
            2,
        )

        # ------------------------------------
        # STEP 2: RETRIEVE PAPER EVIDENCE
        # ------------------------------------

        step = perf_counter()

        # CRITICAL FIX:
        # We now pass the selected paper ID to
        # the retriever instead of ignoring it.
        passages = self.retriever.search(
            question,
            setting=setting,
            paper_id=paper_id,
        )

        record["stages_ms"]["retrieval"] = round(
            (perf_counter() - step) * 1000,
            2,
        )

        record["retrieved_citations"] = [
            passage.citation
            for passage in passages
        ]

        if not passages:

            return finish(
                REFUSAL,
                "ok",
                "refused_no_evidence",
            )

        # ------------------------------------
        # STEP 3: CONSTRUCT GROUNDED PROMPT
        # ------------------------------------

        step = perf_counter()

        messages = make_grounded_messages(
            question,
            passages,
        )

        record["stages_ms"]["prompt_construction"] = (
            round(
                (perf_counter() - step) * 1000,
                2,
            )
        )

        # ------------------------------------
        # STEP 4: MODEL CALL
        # STEP 5: CITATION VALIDATION
        # ------------------------------------

        max_attempts = (
            2
            if allow_revision and self.source == "real"
            else 1
        )

        message = REFUSAL
        outcome = "not_verified"
        final_status = "error"

        for attempt in range(max_attempts):

            call_offset_ms = (
                perf_counter() - started
            ) * 1000

            result = await self._model_call(
                messages
            )

            record["model_calls"].append(
                result
            )

            record["stages_ms"]["model_calls"] = (
                round(
                    sum(
                        call["total_ms"]
                        for call in record["model_calls"]
                    ),
                    2,
                )
            )

            if (
                record["provider_first_content_ms"]
                is None
                and result["first_content_ms"] is not None
            ):

                record["provider_first_content_ms"] = (
                    round(
                        call_offset_ms
                        + result["first_content_ms"],
                        2,
                    )
                )

            final_status = result["status"]

            if final_status != "ok":

                if final_status == "rate_limited":

                    message = (
                        "The research assistant is "
                        "currently rate-limited. "
                        "Please try again later."
                    )

                else:

                    message = (
                        "The model service could not "
                        "complete the request. "
                        "Please try again later."
                    )

                outcome = "model_failure"
                break

            if self.source == "stand-in":

                message = result["text"]
                outcome = "simulation_ungraded"
                break

            # --------------------------------
            # VERIFY CITATION PROVENANCE
            # --------------------------------

            step = perf_counter()

            valid = check_citations(
                result["text"],
                passages,
            )

            refused = looks_like_refusal(
                result["text"]
            )

            record["stages_ms"]["citation_check"] = (
                round(
                    record["stages_ms"].get(
                        "citation_check",
                        0.0,
                    )
                    + (perf_counter() - step) * 1000,
                    2,
                )
            )

            if valid:

                message = normalize_citations(
                    result["text"]
                )

                record["recognized_citations"] = [
                    f"[{paper}, PDF p. {page}]"
                    for paper, page in extract_citations(
                        result["text"]
                    )
                ]

                outcome = "citation_format_checked"
                break

            if refused:

                message = result["text"]
                outcome = "model_refusal_ungraded"
                break

            # --------------------------------
            # OPTIONAL CITATION REPAIR
            # --------------------------------

            if attempt + 1 < max_attempts:

                record["revisions"] += 1

                allowed = ", ".join(
                    sorted({
                        passage.citation
                        for passage in passages
                    })
                )

                messages = messages + [
                    {
                        "role": "assistant",
                        "content": result["text"],
                    },
                    {
                        "role": "user",
                        "content": (
                            "Revise using ONLY supported "
                            "facts and these citations: "
                            + allowed
                            + ". If evidence is insufficient, "
                            "begin with 'I cannot verify'. "
                            "Do not invent citations."
                        ),
                    },
                ]

            else:

                message = REFUSAL
                outcome = "citation_rejected_refused"

        # ------------------------------------
        # AGGREGATE TOKEN USAGE
        # ------------------------------------

        for field in (
            "tokens_in",
            "tokens_out",
        ):

            values = [
                call[field]
                for call in record["model_calls"]
            ]

            if (
                values
                and all(
                    value is not None
                    for value in values
                )
            ):

                record[field] = sum(values)

            else:

                record[field] = None

        return finish(
            message,
            final_status,
            outcome,
        )
