from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

from medical_rag.retrieval.vector_store import (
    DEFAULT_METADATA_FIELDS,
    DEFAULT_MODEL_NAME,
    Tee,
    encode_queries,
    ensure_output_dirs,
    ensure_project_hf_cache,
    human_seconds,
    load_sentence_transformer,
    markdown_table,
    now_iso,
    project_root_from_script,
    resolve_device,
    truncate_query,
)


MEDICAL_QUERIES = [
    "EGFR mutation lung cancer treatment",
    "HIV reverse transcriptase inhibitor resistance",
    "type 2 diabetes insulin sensitivity",
    "SARS coronavirus spike protein",
    "breast cancer gene expression",
    "polymerase chain reaction DNA amplification",
    "inflammatory cytokines macrophage response",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a BGE Chroma vector index.")
    parser.add_argument("--persist_dir", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--n_results", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self_sample_size", type=int, default=20)
    return parser.parse_args()


def get_collection(persist_dir: Path, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(collection_name)


def first_result_rows(query: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for idx, chunk_id in enumerate(ids):
        meta = metas[idx] or {}
        rows.append(
            {
                "query": query,
                "rank": idx + 1,
                "distance": distances[idx] if idx < len(distances) else "",
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "source_title": meta.get("source_title", ""),
                "journal": meta.get("journal", ""),
                "pub_year": meta.get("pub_year", ""),
                "pmcid": meta.get("pmcid", ""),
                "section_title": meta.get("section_title", ""),
                "split_strategy": meta.get("split_strategy", ""),
                "text_preview": (docs[idx] or "")[:800].replace("\n", " "),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_filters(collection, sample_meta: dict[str, Any], n_results: int) -> list[dict[str, Any]]:
    filter_specs: list[tuple[str, dict[str, Any]]] = [
        ("split_strategy_semantic_section", {"split_strategy": "semantic_section"}),
        ("article_type_research_article", {"article_type": "research-article"}),
        ("pub_year_2010_string", {"pub_year": "2010"}),
        ("journal_plos_one", {"journal": "PLoS ONE"}),
    ]
    if sample_meta.get("pub_year"):
        filter_specs.append(("sample_pub_year", {"pub_year": sample_meta["pub_year"]}))
    if sample_meta.get("pmcid"):
        filter_specs.append(("real_pmcid", {"pmcid": sample_meta["pmcid"]}))

    rows = []
    for name, where in filter_specs:
        try:
            result = collection.get(where=where, limit=n_results, include=["metadatas", "documents"])
            count = len(result.get("ids", []))
            rows.append(
                {
                    "filter_name": name,
                    "where": where,
                    "hit_count": count,
                    "status": "ok",
                    "note": "matched" if count else "no matching rows in this collection or test subset",
                }
            )
        except Exception as exc:
            rows.append({"filter_name": name, "where": where, "hit_count": 0, "status": "error", "note": repr(exc)})
    try:
        result = collection.query(query_embeddings=[[0.0] * 768], n_results=1, where={"pmcid": "__NO_SUCH_PMCID__"})
        rows.append(
            {
                "filter_name": "no_result_filter",
                "where": {"pmcid": "__NO_SUCH_PMCID__"},
                "hit_count": len(result.get("ids", [[]])[0]),
                "status": "ok",
                "note": "empty result expected",
            }
        )
    except Exception as exc:
        rows.append({"filter_name": "no_result_filter", "where": {"pmcid": "__NO_SUCH_PMCID__"}, "hit_count": 0, "status": "error", "note": repr(exc)})
    return rows


def make_report(prefix: str, stats: dict[str, Any], query_rows: list[dict[str, Any]], filter_rows: list[dict[str, Any]], self_rows: list[dict[str, Any]]) -> str:
    return f"""# 向量索引质量验证（{prefix}）

## 1. 基础统计

- Collection: `{stats['collection_name']}`
- Persist dir: `{stats['persist_dir']}`
- Collection count: `{stats['collection_count']}`
- Metadata complete in sample: `{stats['metadata_complete_in_sample']}`
- Self top1 hit rate: `{stats['self_top1_hit_rate']}`
- Self top3 same-doc hit rate: `{stats['self_top3_same_doc_hit_rate']}`
- Elapsed: `{human_seconds(float(stats['elapsed_seconds']))}`

## 2. 自相似性验证

{markdown_table(self_rows[:20], ['sample_id', 'chunk_id', 'doc_id', 'top1_chunk_id', 'top1_is_self', 'top3_same_doc'])}

## 3. 医学 Query 检索

{markdown_table(query_rows[:20], ['query', 'rank', 'distance', 'chunk_id', 'doc_id', 'source_title', 'pub_year', 'section_title'])}

## 4. Metadata Filter 验证

{markdown_table(filter_rows, ['filter_name', 'where', 'hit_count', 'status', 'note'])}

## 5. 边界 Query

- 空查询：已在脚本内拦截，不调用模型。
- 超长查询：脚本会压缩空白并截断到 4000 字符再检索。
- 无结果 filter：返回空结果，不视为失败。
"""


def main() -> int:
    args = parse_args()
    project_root = project_root_from_script(__file__)
    ensure_output_dirs(project_root)
    ensure_project_hf_cache(project_root)
    log_path = project_root / "logs" / f"13_validate_chroma_index_{args.output_prefix}.log"
    tee = Tee(log_path)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        start = time.time()
        import pandas as pd

        persist_dir = (project_root / args.persist_dir).resolve()
        collection = get_collection(persist_dir, args.collection_name)
        count = collection.count()
        print(f"[START] {now_iso()} collection_count={count}")
        if count <= 0:
            raise RuntimeError("Collection is empty.")

        sample = collection.get(limit=args.self_sample_size, include=["documents", "metadatas"])
        sample_ids = sample.get("ids", [])
        sample_docs = sample.get("documents", [])
        sample_metas = sample.get("metadatas", [])
        metadata_complete = all(all(field in (meta or {}) for field in DEFAULT_METADATA_FIELDS) for meta in sample_metas)

        device = resolve_device(args.device)
        model = load_sentence_transformer(args.model_name, device, project_root)

        self_rows = []
        top1_hits = 0
        top3_same_doc = 0
        for idx, chunk_id in enumerate(sample_ids):
            doc = sample_docs[idx] or ""
            meta = sample_metas[idx] or {}
            query = truncate_query(doc[:500])
            if not query:
                continue
            vector = encode_queries(model, [query], batch_size=1)[0]
            result = collection.query(query_embeddings=[vector], n_results=max(args.n_results, 3), include=["metadatas", "documents", "distances"])
            ids = result.get("ids", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            top1 = ids[0] if ids else ""
            same_doc = any((m or {}).get("doc_id") == meta.get("doc_id") for m in metas[:3])
            is_self = top1 == chunk_id
            top1_hits += int(is_self)
            top3_same_doc += int(same_doc)
            self_rows.append(
                {
                    "sample_id": idx,
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id", ""),
                    "top1_chunk_id": top1,
                    "top1_is_self": is_self,
                    "top3_same_doc": same_doc,
                }
            )

        query_rows: list[dict[str, Any]] = []
        for query in MEDICAL_QUERIES:
            vector = encode_queries(model, [query], batch_size=1)[0]
            result = collection.query(query_embeddings=[vector], n_results=args.n_results, include=["metadatas", "documents", "distances"])
            query_rows.extend(first_result_rows(query, result))

        empty_query_intercepted = True
        long_query = truncate_query("EGFR " * 3000)
        vector = encode_queries(model, [long_query], batch_size=1)[0]
        collection.query(query_embeddings=[vector], n_results=1)

        sample_meta = sample_metas[0] if sample_metas else {}
        filter_rows = validate_filters(collection, sample_meta or {}, args.n_results)

        query_path = project_root / "artifacts/metrics/t008_vector_index" / f"vector_query_results_{args.output_prefix}.csv"
        filter_path = project_root / "artifacts/metrics/t008_vector_index" / f"vector_filter_validation_{args.output_prefix}.csv"
        self_path = project_root / "artifacts/metrics/t008_vector_index" / f"vector_self_similarity_{args.output_prefix}.csv"
        report_path = project_root / "reports/formal" / f"向量索引质量验证_{args.output_prefix}.md"

        query_fields = ["query", "rank", "distance", "chunk_id", "doc_id", "source_title", "journal", "pub_year", "pmcid", "section_title", "split_strategy", "text_preview"]
        write_csv(query_path, query_rows, query_fields)
        write_csv(filter_path, filter_rows, ["filter_name", "where", "hit_count", "status", "note"])
        write_csv(self_path, self_rows, ["sample_id", "chunk_id", "doc_id", "top1_chunk_id", "top1_is_self", "top3_same_doc"])

        denominator = max(len(self_rows), 1)
        stats = {
            "collection_name": args.collection_name,
            "persist_dir": str(persist_dir),
            "collection_count": count,
            "metadata_complete_in_sample": metadata_complete,
            "self_top1_hit_rate": round(top1_hits / denominator, 4),
            "self_top3_same_doc_hit_rate": round(top3_same_doc / denominator, 4),
            "empty_query_intercepted": empty_query_intercepted,
            "elapsed_seconds": round(time.time() - start, 3),
        }
        report_path.write_text(make_report(args.output_prefix, stats, query_rows, filter_rows, self_rows), encoding="utf-8")
        print(f"[DONE] report={report_path}")
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
