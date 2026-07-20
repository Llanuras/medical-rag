from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    texts = payload["texts"]
    emb = HuggingFaceEmbeddings(model_name=MODEL, model_kwargs={"device": "cpu"}, encode_kwargs={"batch_size": 1})
    vectors = []
    for text in texts:
        vectors.append(emb.embed_query(text))
    Path(args.output).write_text(json.dumps({"model": MODEL, "embeddings": vectors}), encoding="utf-8")
    print(f"embedded {len(vectors)} texts with {MODEL}")

if __name__ == "__main__":
    main()
