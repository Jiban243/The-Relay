
"""Update The Relay PDF with the AI collaboration history.

Run:
    python -m scripts.patch_submission_history

Preserves original tables and measurements.
Creates a revised two-page PDF.
"""

from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "SUBMISSION_FINAL_2PAGES.pdf"

TARGET = ROOT / "SUBMISSION_FINAL_2PAGES_REVISED.pdf"

CHAT_URL = (
    "https://chatgpt.com/share/"
    "6af4c906-e13c-83e8-b604-4eb8689ddfc5"
)


HTML = f"""
<h3>How I worked with the assistant</h3>

<p><b>Three prompts that mattered:</b></p>

<p>
<b>1.</b> "Be thorough, make no assumptions" -
fixed the scope and required checks before implementing.
</p>

<p>
<b>2.</b> "Don't give download version, give copy paste
one always" - made every change inspectable and
testable locally.
</p>

<p>
<b>3.</b> "Check if the required answers are retrieved
and produced to the correct question; if absent in
the 3 papers, say not present" - made grounding
and refusal the acceptance criterion.
</p>

<p>
<b>Mistake caught:</b> The Transformer encoder/decoder
question retrieved H-Net passages, producing false
refusals. I traced it to ignored paper scope and
verified the retrieval fix with regression tests.
</p>

<p>
<b>Decision overruled:</b> I rejected the assistant's
45-step manual grading proposal. We used an auditable
offline rubric (6 human, 39 rubric grades) and
disclosed that it is not a full factual audit.
</p>

<p>
<b>My contribution:</b> I specified constraints,
ran and checked experiments, challenged wrong
results, chose trade-offs and verified the
223-request dataset. AI assisted with implementation.
</p>

<p>
<b>Conversation history link:</b>
<a href="{CHAT_URL}">{CHAT_URL}</a>
</p>
"""


CSS = """
* {
    font-family: Helvetica;
    font-size: 8.7pt;
    color: #17324d;
}

h3 {
    font-size: 10.1pt;
    font-weight: bold;
    margin: 0 0 5pt 0;
}

p {
    margin: 0 0 4.6pt 0;
    line-height: 1.13;
}

a {
    color: #155e95;
    text-decoration: underline;
}
"""


def main():

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing original PDF: {SOURCE}"
        )

    with pymupdf.open(SOURCE) as document:

        if len(document) != 2:
            raise RuntimeError(
                "Expected exactly two pages"
            )

        original_page_1 = document[0].get_text()

        page = document[1]

        matches = page.search_for(
            "How I worked with the assistant"
        )

        if len(matches) != 1:
            raise RuntimeError(
                "Could not uniquely locate the "
                "collaboration section"
            )

        heading = matches[0]

        if (
            heading.x0 <= page.rect.width / 2
            or heading.y0 < 250
            or heading.y0 > 440
        ):
            raise RuntimeError(
                "Unexpected PDF layout: "
                f"{heading}"
            )

        # Replace only the existing collaboration
        # section in the right-hand column.
        region = pymupdf.Rect(
            heading.x0 - 5,
            heading.y0 - 5,
            page.rect.width - 26,
            page.rect.height - 43,
        )

        if region.height < 120:
            raise RuntimeError(
                "Insufficient space for revised section"
            )

        # Ensure we do not erase text from another column.
        for block in page.get_text("blocks"):

            if len(block) < 7 or block[6] != 0:
                continue

            block_rect = pymupdf.Rect(block[:4])

            if (block_rect & region).is_empty:
                continue

            if block_rect.x0 < page.rect.width / 2:
                raise RuntimeError(
                    "Edit area overlaps another section"
                )

        page.add_redact_annot(
            region,
            fill=(1, 1, 1),
            cross_out=False,
        )

        page.apply_redactions(
            images=0,
            graphics=0,
            text=0,
        )

        remaining, scale = page.insert_htmlbox(
            region,
            HTML,
            css=CSS,
            scale_low=0.7,
        )

        if remaining < 0:
            raise RuntimeError(
                "Updated section does not fit"
            )

        if scale < 0.85:
            raise RuntimeError(
                "Text would be too small to read"
            )

        document.save(
            TARGET,
            garbage=4,
            deflate=True,
        )

    # Verify the output.
    with pymupdf.open(TARGET) as pdf:

        assert len(pdf) == 2

        if pdf[0].get_text() != original_page_1:
            raise RuntimeError(
                "Page 1 changed unexpectedly"
            )

        text = pdf[1].get_text()

        required = [
            "How I worked with the assistant",
            "Mistake caught",
            "Decision overruled",
            "My contribution",
            "Conversation history link",
            "Retrieval: quality versus cost",
            "Your feedback on Vaani",
        ]

        missing = [
            item
            for item in required
            if item not in text
        ]

        if missing:
            raise RuntimeError(
                f"Missing content: {missing}"
            )

        links = [
            link.get("uri")
            for link in pdf[1].get_links()
        ]

        if CHAT_URL not in links:
            raise RuntimeError(
                "Clickable conversation link missing"
            )

        for page in pdf:

            for block in page.get_text("blocks"):

                if len(block) < 7 or block[6] != 0:
                    continue

                x0, y0, x1, y1 = block[:4]

                if (
                    x0 < 0
                    or y0 < 0
                    or x1 > page.rect.width + 0.5
                    or y1 > page.rect.height + 0.5
                ):
                    raise RuntimeError(
                        "Text extends outside page boundaries"
                    )

    print("\n========== UPDATED SUBMISSION ==========")

    print("Created:", TARGET)
    print("Pages: 2")
    print("Original page 1: preserved")
    print("Other page 2 sections: preserved")
    print("Collaboration section: expanded")
    print("Clickable conversation link: verified")
    print("Original PDF: unchanged")


if __name__ == "__main__":
    main()
