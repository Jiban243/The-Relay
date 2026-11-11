
from pathlib import Path
import hashlib
import json
import pymupdf

PAPERS = {
    "1706.03762v7": "attention_v7.pdf",
    "2410.00037v1": "moshi_v1.pdf",
    "2507.07955v2": "dynamic_chunking_v2.pdf",
}

paper_dir = Path("data/papers")
manifest = {}

for paper_id, filename in PAPERS.items():
    path = paper_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    pdf_bytes = path.read_bytes()

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError(f"Not a valid PDF header: {filename}")

    with pymupdf.open(path) as doc:
        pages = len(doc)
        first_page = doc[0].get_text("text")
        total_chars = sum(len(page.get_text("text")) for page in doc)

    if pages == 0 or total_chars == 0:
        raise ValueError(f"Empty or unreadable PDF: {filename}")

    manifest[paper_id] = {
        "filename": filename,
        "source_url": f"https://arxiv.org/pdf/{paper_id}",
        "pages": pages,
        "text_characters": total_chars,
        "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
    }

    print(f"\nPaper: {paper_id}")
    print(f"File: {filename}")
    print(f"Pages: {pages}")
    print(f"Extracted characters: {total_chars}")
    print(f"First-page preview: {' '.join(first_page.split())[:150]}")

manifest_path = paper_dir / "manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print(f"\nVerification complete. Manifest: {manifest_path}")
