from __future__ import annotations

import argparse
from pathlib import Path

from medical_rag.retrieval.vector_store import (
    DEFAULT_MODEL_NAME,
    encode_queries,
    ensure_project_hf_cache,
    load_sentence_transformer,
    parse_where_json,
    project_root_from_script,
    resolve_device,
    truncate_query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a persistent Chroma index with BGE query instruction.")
    parser.add_argument("--persist_dir", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--query", required=True)
    parser.add_argument("--n_results", type=int, default=5)
    parser.add_argument("--where_json", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = truncate_query(args.query)
    if not query:
        print("Empty query is not allowed.")
        return 2

    import chromadb

    project_root = project_root_from_script(__file__)
    ensure_project_hf_cache(project_root)
    persist_dir = (project_root / args.persist_dir).resolve()
    where = parse_where_json(args.where_json)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(args.collection_name)
    device = resolve_device(args.device)
    model = load_sentence_transformer(args.model_name, device, project_root)
    vector = encode_queries(model, [query], batch_size=1)[0]
    result = collection.query(
        query_embeddings=[vector],
        n_results=args.n_results,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    if not ids:
        print("No results.")
        return 0

    for idx, chunk_id in enumerate(ids):
        meta = metas[idx] or {}
        distance = distances[idx] if idx < len(distances) else ""
        preview = (docs[idx] or "")[:800].replace("\n", " ")
        print(f"\nRank: {idx + 1}")
        print(f"distance: {distance}")
        print(f"chunk_id: {chunk_id}")
        print(f"source_title: {meta.get('source_title', '')}")
        print(f"journal: {meta.get('journal', '')}")
        print(f"pub_year: {meta.get('pub_year', '')}")
        print(f"pmcid: {meta.get('pmcid', '')}")
        print(f"section_title: {meta.get('section_title', '')}")
        print(f"text_preview: {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
