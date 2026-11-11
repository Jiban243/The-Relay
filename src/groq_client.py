
from dataclasses import dataclass, field
from time import perf_counter
import os

from dotenv import load_dotenv

from groq import (
    AsyncGroq,
    RateLimitError,
    APITimeoutError,
    APIStatusError,
    APIConnectionError,
)


@dataclass
class ModelResult:
    text: str
    status: str
    first_token_ms: float | None
    total_ms: float

    tokens_in: int | None = None
    tokens_out: int | None = None

    http_status: int | None = None
    retry_after: str | None = None
    error_type: str | None = None

    headers: dict = field(default_factory=dict)


class GroqModelClient:

    def __init__(
        self,
        model=None,
        timeout_seconds=30,
    ):

        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. "
                "Create a .env file first."
            )

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

        self.client = AsyncGroq(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def complete(
        self,
        messages,
        max_output_tokens=320,
        on_piece=None,
    ):

        start = perf_counter()

        first_token_ms = None
        pieces = []

        tokens_in = None
        tokens_out = None

        try:

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=0.1,
                stream=True,
            )

            async for event in stream:

                choices = getattr(event, "choices", [])

                if choices:

                    delta = getattr(
                        choices[0], "delta", None
                    )

                    piece = (
                        getattr(delta, "content", None)
                        if delta is not None
                        else None
                    )

                    if piece:

                        if first_token_ms is None:
                            first_token_ms = (
                                perf_counter() - start
                            ) * 1000

                        pieces.append(piece)

                        if on_piece:
                            on_piece(piece)

                # Some Groq streaming responses provide
                # exact usage on the final event.
                usage = getattr(event, "usage", None)

                if usage is None:
                    x_groq = getattr(
                        event, "x_groq", None
                    )

                    if x_groq is not None:
                        usage = getattr(
                            x_groq, "usage", None
                        )

                if usage is not None:

                    tokens_in = getattr(
                        usage, "prompt_tokens", None
                    )

                    tokens_out = getattr(
                        usage, "completion_tokens", None
                    )

            total_ms = (
                perf_counter() - start
            ) * 1000

            return ModelResult(
                text="".join(pieces),
                status="ok",
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        except RateLimitError as error:
            return self.handle_error(
                error,
                start,
                "rate_limited",
                first_token_ms,
                "".join(pieces),
            )

        except APITimeoutError as error:
            return self.handle_error(
                error,
                start,
                "timeout",
                first_token_ms,
                "".join(pieces),
            )

        except (
            APIStatusError,
            APIConnectionError,
        ) as error:

            return self.handle_error(
                error,
                start,
                "error",
                first_token_ms,
                "".join(pieces),
            )

    @staticmethod
    def handle_error(
        error,
        start,
        status,
        first_token_ms,
        partial_text,
    ):

        response = getattr(
            error, "response", None
        )

        headers = {}

        if response is not None:
            headers = dict(response.headers)

        selected_headers = {
            key: value
            for key, value in headers.items()
            if (
                key.lower().startswith("x-ratelimit")
                or key.lower() == "retry-after"
            )
        }

        return ModelResult(
            text=partial_text,
            status=status,
            first_token_ms=first_token_ms,
            total_ms=(
                perf_counter() - start
            ) * 1000,
            http_status=getattr(
                error, "status_code", None
            ),
            retry_after=headers.get("retry-after"),
            error_type=type(error).__name__,
            headers=selected_headers,
        )

    async def close(self):
        await self.client.close()
