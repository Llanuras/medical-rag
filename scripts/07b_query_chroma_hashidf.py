from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.config import Settings

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?")

DEFAULT_QUERIES = [
    "Plasmodium falciparum intraerythrocytic developmental cycle transcriptome",
    "pRb inactivation mammary cells tumor initiation progression",
    "type 2 diabetes high protein diet insulin concentration",
    "immune response B cells master regulator",
    "SARS coronavirus spike protein trafficking",
]


class HashingTfidfEmbeddingFunction:
    def __init__(self, texts: list[str], dim: int = 384):
        self.dim = dim
        doc_freq: Counter[int] = Counter()
        for text in texts:
            doc_freq.update({self._index(token) for token in TOKEN_RE.findall((text or "").lower())})
        doc_count = max(1, len(texts))
        self.idf = {
            index: math.log((1 + doc_count) / (1 + freq)) + 1.0
            for index, freq in doc_freq.items()
        }

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False) % self.dim

    def embed_query(self, text: str) -> list[float]:
        counts: Counter[int] = Counter()
        for token in TOKEN_RE.findall((text or "").lower()):
            counts[self._index(token)] += 1
        vector = [0.0] * self.dim
        for index, count in counts.items():
            vector[index] = float((1 + math.log(count)) * self.idf.get(index, 1.0))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [float(value / norm) for value in vector]
        return vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="archive/experiments/indexes/chroma_fulltext_limit3028_hashidf")
    parser.add_argument("--collection", default="pmc_fulltext_scale")
    parser.add_argument("--output", default="artifacts/metrics/t002_corpus_analysis/fulltext_chroma_query_results_limit3028_hashidf_targeted.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    client = chromadb.PersistentClient(
        path=args.persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(args.collection)
    payload = collection.get(include=["documents"])
    docs = payload["documents"]
    embedding = HashingTfidfEmbeddingFunction(docs, dim=384)

    rows = []
    for query in args.query or DEFAULT_QUERIES:
        result = collection.query(query_embeddings=[embedding.embed_query(query)], n_results=args.top_k)
        for rank, (doc_id, text, meta, distance) in enumerate(
            zip(
                result.get("ids", [[]])[0],
                result.get("documents", [[]])[0],
                result.get("metadatas", [[]])[0],
                result.get("distances", [[]])[0],
            ),
            start=1,
        ):
            rows.append(
                {
                    "query": query,
                    "rank": rank,
                    "distance": distance,
                    "id": doc_id,
                    "record_id": meta.get("record_id", ""),
                    "title": meta.get("title", ""),
                    "section_title": meta.get("section_title", ""),
                    "source_file": meta.get("source_file", ""),
                    "snippet": " ".join((text or "").split())[:320],
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", out)


if __name__ == "__main__":
    main()
