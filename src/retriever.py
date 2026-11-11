
from dataclasses import dataclass, replace
from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi

from .query_router import route_question


STOPWORDS = set("""
a an and are as at be by can could did do does
for from had has have how in is it its of on or
our that the their there these this to was were
what when where which who why will with would
you your explain describe according paper authors
""".split())


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())

    return [
        word for word in words
        if word not in STOPWORDS
    ]


@dataclass(frozen=True)
class RetrievalSetting:
    name: str
    chunk_words: int
    top_k: int
    overlap_words: int
    rerank: bool = False

    def __post_init__(self):
        if self.chunk_words < 30:
            raise ValueError("Chunk size must be at least 30 words")

        if self.top_k < 1:
            raise ValueError("top_k must be positive")

        if not 0 <= self.overlap_words < self.chunk_words:
            raise ValueError("Invalid chunk overlap")


SETTINGS = {
    "small": RetrievalSetting(
        name="small",
        chunk_words=250,
        top_k=2,
        overlap_words=35,
    ),

    "balanced": RetrievalSetting(
        name="balanced",
        chunk_words=500,
        top_k=4,
        overlap_words=50,
    ),

    "reranked": RetrievalSetting(
        name="reranked",
        chunk_words=500,
        top_k=4,
        overlap_words=50,
        rerank=True,
    ),
}


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    paper_id: str
    title: str
    pdf_page: int
    source_url: str
    text: str
    score: float = 0.0

    @property
    def citation(self):
        return f"[{self.paper_id}, PDF p. {self.pdf_page}]"


class BM25Retriever:

    def __init__(self, pages_path):

        self.pages_path = Path(pages_path)

        if not self.pages_path.exists():
            raise FileNotFoundError(self.pages_path)

        self.pages = []

        with self.pages_path.open(
            "r", encoding="utf-8"
        ) as file:

            for line in file:
                if line.strip():
                    self.pages.append(json.loads(line))

        if not self.pages:
            raise ValueError("No extracted pages found")

        self.indexes = {}

    def build_index(self, setting):

        passages = []

        for page in self.pages:

            words = page["text"].split()

            chunk_size = setting.chunk_words
            stride = chunk_size - setting.overlap_words

            for start in range(0, len(words), stride):

                chunk_words = words[start:start + chunk_size]

                if not chunk_words:
                    continue

                chunk = Passage(
                    chunk_id=(
                        f"{page['paper_id']}:"
                        f"p{page['page']}:w{start}"
                    ),
                    paper_id=page["paper_id"],
                    title=page["title"],
                    pdf_page=page["page"],
                    source_url=page["source_url"],
                    text=" ".join(chunk_words),
                )

                passages.append(chunk)

                if start + chunk_size >= len(words):
                    break

        tokenized_corpus = [
            tokenize(p.text) or ["_empty_"]
            for p in passages
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        self.indexes[setting] = {
            "passages": passages,
            "bm25": bm25,
            "tokens": tokenized_corpus,
        }

    def search(
        self,
        question,
        setting="balanced",
        paper_id=None,
    ):

        if isinstance(setting, str):
            setting = SETTINGS[setting]

        if setting not in self.indexes:
            self.build_index(setting)

        index = self.indexes[setting]

        passages = index["passages"]
        bm25 = index["bm25"]
        document_tokens = index["tokens"]

        route = route_question(question)

        if paper_id is None:
            paper_id = route.paper_id

        query_tokens = tokenize(route.search_query)

        if not query_tokens:
            return []

        query_terms = set(query_tokens)
        scores = bm25.get_scores(query_tokens)

        candidates = []

        for i, passage in enumerate(passages):

            if paper_id and passage.paper_id != paper_id:
                continue

            overlap = len(
                query_terms.intersection(document_tokens[i])
            )

            if overlap == 0:
                continue

            candidates.append((i, float(scores[i]), overlap))

        # First-stage BM25 ranking.
        candidates.sort(
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )

        if setting.rerank:

            # Deterministic lexical re-ranking.
            # Fetch more BM25 candidates, then use coverage.
            candidates = candidates[:max(setting.top_k * 5, 20)]

            candidates.sort(
                key=lambda x: (
                    x[2] / len(query_terms),
                    x[1],
                ),
                reverse=True,
            )

        selected = candidates[:setting.top_k]

        return [
            replace(passages[i], score=score)
            for i, score, overlap in selected
        ]

    def chunk_count(self, setting="balanced"):

        if isinstance(setting, str):
            setting = SETTINGS[setting]

        if setting not in self.indexes:
            self.build_index(setting)

        return len(self.indexes[setting]["passages"])
