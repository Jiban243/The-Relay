
from pathlib import Path
import json
import pymupdf

ROOT = Path(__file__).resolve().parents[1]

PAPERS_DIR = ROOT / "data" / "papers"
OUTPUT_DIR = ROOT / "data" / "processed"

MANIFEST_PATH = PAPERS_DIR / "manifest.json"
OUTPUT_PATH = OUTPUT_DIR / "pages.jsonl"

TITLES = {
    "1706.03762v7": "Attention Is All You Need",
    "2410.00037v1": "Moshi: a speech-text foundation model for real-time dialogue",
    "2507.07955v2": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
}


def extract_papers():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    total_pages = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for paper_id, metadata in manifest.items():

            pdf_path = PAPERS_DIR / metadata["filename"]

            print(f"\nProcessing: {paper_id}")

            with pymupdf.open(pdf_path) as pdf:

                if len(pdf) != metadata["pages"]:
                    raise ValueError(
                        f"Page count changed for {paper_id}"
                    )

                for page_number, page in enumerate(pdf, start=1):

                    text = page.get_text("text", sort=True).strip()

                    record = {
                        "paper_id": paper_id,
                        "title": TITLES[paper_id],
                        "page": page_number,
                        "source_url": metadata["source_url"],
                        "text": text,
                    }

                    output.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                    if not text:
                        print(f"Warning: empty page {page_number}")

                    total_pages += 1

                print(f"Extracted {len(pdf)} pages")

    print("\nExtraction complete!")
    print(f"Total pages: {total_pages}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_papers()
