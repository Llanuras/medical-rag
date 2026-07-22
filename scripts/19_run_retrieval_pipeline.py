from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_rag.retrieval.multipath import DEFAULT_BM25_DIR, DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DIR
from medical_rag.retrieval.pipeline import RetrievalPipeline
from medical_rag.retrieval.reranker import DEFAULT_RERANKER_MODEL
from medical_rag.retrieval.vector_store import DEFAULT_MODEL_NAME, Tee, now_iso, parse_where_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run query understanding, multi-path retrieval, optional reranking, and scoring.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k_vector", type=int, default=50)
    parser.add_argument("--top_k_keyword", type=int, default=50)
    parser.add_argument("--fusion_strategy", choices=["rrf", "weighted", "simple"], default="rrf")
    parser.add_argument("--fusion_top_k", type=int, default=80)
    parser.add_argument("--rerank_top_k", type=int, default=50)
    parser.add_argument("--final_top_k", type=int, default=10)
    parser.add_argument("--chroma_persist_dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--chroma_collection_name", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument("--bm25_index_dir", default=str(DEFAULT_BM25_DIR))
    parser.add_argument("--embedding_model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker_model_name", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--disable_reranker", action="store_true")
    parser.add_argument("--where_json", default=None)
    parser.add_argument("--terminology_path", default=None)
    parser.add_argument("--term_index_path", default=None)
    parser.add_argument("--reranker_batch_size", type=int, default=8)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--vector_weight", type=float, default=0.65)
    parser.add_argument("--keyword_weight", type=float, default=0.35)
    parser.add_argument("--rrf_k", type=int, default=60)
    return parser.parse_args()


def _validate_output_prefix(prefix: str) -> str:
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("output_prefix must be a non-empty filename-safe scope without path separators")
    return prefix


def _result_row(query: str, item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    return {
        "query": query,
        "final_rank": item.get("final_rank"),
        "chunk_id": item.get("chunk_id"),
        "doc_id": item.get("doc_id"),
        "source_title": metadata.get("source_title"),
        "journal": metadata.get("journal"),
        "pub_year": metadata.get("pub_year"),
        "pmid": metadata.get("pmid"),
        "pmcid": metadata.get("pmcid"),
        "section_title": metadata.get("section_title"),
        "retrieval_sources": ",".join(item.get("retrieval_sources") or []),
        "vector_rank": item.get("vector_rank"),
        "keyword_rank": item.get("keyword_rank"),
        "vector_score": item.get("vector_score"),
        "vector_distance": item.get("vector_distance"),
        "bm25_score": item.get("bm25_score"),
        "fusion_score": item.get("fusion_score"),
        "fusion_strategy": item.get("fusion_strategy"),
        "reranker_raw_score": item.get("reranker_raw_score"),
        "reranker_rank": item.get("reranker_rank"),
        "relevance_score": item.get("relevance_score"),
        "recency_score": item.get("recency_score"),
        "authority_score": item.get("authority_score"),
        "final_score": item.get("final_score"),
        "text": item.get("text"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(_result_row("", {}).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(result: dict[str, Any], args: argparse.Namespace) -> str:
    evidence = result.get("evidence", [])
    rows = []
    for item in evidence:
        metadata = item.get("metadata") or {}
        rows.append(
            f"| {item.get('final_rank')} | {metadata.get('source_title', '')} | "
            f"{metadata.get('journal', '')} | {metadata.get('pub_year', '')} | "
            f"{item.get('retrieval_sources', [])} | {float(item.get('final_score') or 0):.4f} |"
        )
    table = "\n".join(rows) if rows else "| - | No evidence | - | - | - | - |"
    warnings = "\n".join(f"- {warning}" for warning in result.get("warnings", [])) or "- none"
    return f"""# 完整检索流水线报告（{args.output_prefix}）

## 查询与状态

- Query: `{args.query}`
- Status: `{result.get('status')}`
- Vector candidates: `{result.get('retrieval', {}).get('vector_result_count', 0)}`
- Keyword candidates: `{result.get('retrieval', {}).get('keyword_result_count', 0)}`
- Fused candidates: `{result.get('candidate_count', 0)}`
- Reranker requested: `{result.get('reranker_requested')}`
- Reranker used: `{result.get('reranker_used')}`

## 融合策略说明

- `simple`：按向量结果、BM25 结果的输入顺序合并并按 `chunk_id` 去重，仅作基线。
- `rrf`：使用倒数排名融合，不直接混合 Chroma 与 BM25 的异构原始分数，是当前默认推荐。
- `weighted`：对单次查询的两路分数分别 Min-Max 归一化后加权；向量/BM25 权重 `{args.vector_weight:.2f}/{args.keyword_weight:.2f}` 只是启发式初始值，不是固定标准。

## 多准则排序说明

- 相关性是主信号；reranker 禁用或不可用时，使用归一化 fusion score 降级。
- authority score 是可审计的 journal prior，不是影响因子。
- recency score 只做软排序，不做硬过滤；当前 deprecated 语料年份整体偏旧，不应把该分数解读为文献质量结论。

## Evidence Top-{len(evidence)}

| Rank | Title | Journal | Year | Sources | Final score |
| --- | --- | --- | --- | --- | --- |
{table}

## Warnings

{warnings}

## 范围边界

当前阶段只输出 retrieval evidence list，不生成医学答案。
"""


def main() -> int:
    args = parse_args()
    try:
        _validate_output_prefix(args.output_prefix)
        where = parse_where_json(args.where_json)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2

    metrics_dir = PROJECT_ROOT / "artifacts/metrics/t014_multipath_retrieval"
    reports_dir = PROJECT_ROOT / "reports/formal"
    logs_dir = PROJECT_ROOT / "logs"
    for path in (metrics_dir, reports_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / f"retrieval_pipeline_results_{args.output_prefix}.csv"
    jsonl_path = metrics_dir / f"retrieval_pipeline_results_{args.output_prefix}.jsonl"
    report_path = reports_dir / f"完整检索流水线报告_{args.output_prefix}.md"
    log_path = logs_dir / f"19_run_retrieval_pipeline_{args.output_prefix}.log"
    tee = Tee(log_path)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        print(f"[START] {now_iso()}")
        pipeline = RetrievalPipeline(
            project_root=PROJECT_ROOT,
            chroma_persist_dir=args.chroma_persist_dir,
            chroma_collection_name=args.chroma_collection_name,
            bm25_index_dir=args.bm25_index_dir,
            embedding_model_name=args.embedding_model_name,
            reranker_model_name=args.reranker_model_name,
            terminology_path=args.terminology_path,
            term_index_path=args.term_index_path,
            device=args.device,
            reranker_batch_size=args.reranker_batch_size,
            reranker_max_length=args.reranker_max_length,
            vector_weight=args.vector_weight,
            keyword_weight=args.keyword_weight,
            rrf_k=args.rrf_k,
        )
        result = pipeline.run(
            args.query,
            top_k_vector=args.top_k_vector,
            top_k_keyword=args.top_k_keyword,
            fusion_strategy=args.fusion_strategy,
            fusion_top_k=args.fusion_top_k,
            rerank_top_k=args.rerank_top_k,
            final_top_k=args.final_top_k,
            disable_reranker=args.disable_reranker,
            where=where,
        )
        rows = [_result_row(args.query, item) for item in result.get("evidence", [])]
        _write_csv(csv_path, rows)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for item in result.get("evidence", []):
                payload = {
                    "query": args.query,
                    "query_info": result.get("query_info"),
                    "evidence": item,
                    "warnings": result.get("warnings", []),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        report_path.write_text(make_report(result, args), encoding="utf-8")
        print(f"[DONE] status={result.get('status')} evidence={len(rows)} reranker_used={result.get('reranker_used')}")
        for warning in result.get("warnings", []):
            print(f"[WARNING] {warning}")
        print(f"[CSV] {csv_path}")
        print(f"[JSONL] {jsonl_path}")
        print(f"[REPORT] {report_path}")
        print(f"[LOG] {log_path}")
        return 0 if result.get("status") in {"ok", "no_evidence"} else 2
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
