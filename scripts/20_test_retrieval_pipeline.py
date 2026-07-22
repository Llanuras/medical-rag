from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_rag.retrieval.multipath import DEFAULT_BM25_DIR, DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DIR
from medical_rag.retrieval.pipeline import RetrievalPipeline
from medical_rag.retrieval.reranker import DEFAULT_RERANKER_MODEL
from medical_rag.retrieval.vector_store import DEFAULT_MODEL_NAME, Tee, now_iso


BENCHMARK_QUERIES = [
    "metformin cardiovascular disease",
    "myocardial infarction heart attack",
    "EGFR lung cancer",
    "PCR DNA amplification",
    "SARS coronavirus spike protein",
    "HIV reverse transcriptase inhibitor resistance",
    "breast cancer gene expression",
    "inflammatory cytokines macrophage response",
    "二甲双胍对心血管疾病有何影响？",
    "阿司匹林和心肌梗死有什么关系？",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed T014 medical retrieval benchmark and save top evidence.")
    parser.add_argument("--chroma_persist_dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--chroma_collection_name", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument("--bm25_index_dir", default=str(DEFAULT_BM25_DIR))
    parser.add_argument("--embedding_model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker_model_name", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--disable_reranker", action="store_true")
    parser.add_argument("--fusion_strategy", choices=["rrf", "weighted", "simple"], default="rrf")
    parser.add_argument("--top_k_vector", type=int, default=50)
    parser.add_argument("--top_k_keyword", type=int, default=50)
    parser.add_argument("--fusion_top_k", type=int, default=80)
    parser.add_argument("--rerank_top_k", type=int, default=50)
    parser.add_argument("--final_top_k", type=int, default=10)
    parser.add_argument("--max_queries", type=int, default=None)
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
    result_count: int,
) -> str:
    summary_rows = "\n".join(
        f"| {row['query_id']} | {row['query']} | {row['status']} | {row['evidence_count']} | "
        f"{row['reranker_used']} | {row['elapsed_seconds']:.3f} |"
        for row in summaries
    )
    return f"""# 多路检索与重排序质量验证报告（{args.output_prefix}）

## 范围

- Queries: `{len(summaries)}`
- Saved evidence rows: `{result_count}`
- Fusion strategy: `{args.fusion_strategy}`
- Reranker disabled: `{args.disable_reranker}`
- Final top-k: `{args.final_top_k}`

| ID | Query | Status | Evidence | Reranker used | Seconds |
| --- | --- | --- | --- | --- | --- |
{summary_rows}

## 策略解释

- `simple` 只按输入顺序合并去重，作为基线。
- `rrf` 基于名次做倒数排名融合，是当前默认推荐。
- `weighted` 分别归一化两路分数后使用 `{args.vector_weight:.2f}/{args.keyword_weight:.2f}` 加权；该权重是启发式初始值，不是固定标准。
- authority score 只是 journal prior，不是影响因子。
- recency score 对当前年份偏旧的 deprecated 语料只做软排序，不做硬过滤。
- 本报告记录检索结果，不实施答案生成，也不在没有人工相关性标注时声称 Recall/MRR/nDCG 结论。
"""


def main() -> int:
    args = parse_args()
    try:
        _validate_output_prefix(args.output_prefix)
        if args.max_queries is not None and args.max_queries <= 0:
            raise ValueError("max_queries must be positive when provided")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2

    metrics_dir = PROJECT_ROOT / "artifacts/metrics/t014_multipath_retrieval"
    reports_dir = PROJECT_ROOT / "reports/formal"
    logs_dir = PROJECT_ROOT / "logs"
    for path in (metrics_dir, reports_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    results_path = metrics_dir / f"retrieval_pipeline_benchmark_results_{args.output_prefix}.csv"
    summary_path = metrics_dir / f"retrieval_pipeline_benchmark_summary_{args.output_prefix}.csv"
    report_path = reports_dir / f"多路检索与重排序质量验证报告_{args.output_prefix}.md"
    log_path = logs_dir / f"20_test_retrieval_pipeline_{args.output_prefix}.log"
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
        queries = BENCHMARK_QUERIES[: args.max_queries] if args.max_queries else BENCHMARK_QUERIES
        evidence_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for query_id, query in enumerate(queries, start=1):
            started = time.time()
            result = pipeline.run(
                query,
                top_k_vector=args.top_k_vector,
                top_k_keyword=args.top_k_keyword,
                fusion_strategy=args.fusion_strategy,
                fusion_top_k=args.fusion_top_k,
                rerank_top_k=args.rerank_top_k,
                final_top_k=args.final_top_k,
                disable_reranker=args.disable_reranker,
            )
            elapsed = time.time() - started
            for item in result.get("evidence", []):
                metadata = item.get("metadata") or {}
                evidence_rows.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "final_rank": item.get("final_rank"),
                        "chunk_id": item.get("chunk_id"),
                        "doc_id": item.get("doc_id"),
                        "source_title": metadata.get("source_title"),
                        "journal": metadata.get("journal"),
                        "pub_year": metadata.get("pub_year"),
                        "retrieval_sources": ",".join(item.get("retrieval_sources") or []),
                        "vector_rank": item.get("vector_rank"),
                        "keyword_rank": item.get("keyword_rank"),
                        "fusion_score": item.get("fusion_score"),
                        "relevance_score": item.get("relevance_score"),
                        "recency_score": item.get("recency_score"),
                        "authority_score": item.get("authority_score"),
                        "final_score": item.get("final_score"),
                        "text": item.get("text"),
                    }
                )
            summaries.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "status": result.get("status"),
                    "vector_result_count": result.get("retrieval", {}).get("vector_result_count", 0),
                    "keyword_result_count": result.get("retrieval", {}).get("keyword_result_count", 0),
                    "candidate_count": result.get("candidate_count", 0),
                    "evidence_count": len(result.get("evidence", [])),
                    "reranker_used": result.get("reranker_used"),
                    "warning_count": len(result.get("warnings", [])),
                    "warnings_json": json.dumps(result.get("warnings", []), ensure_ascii=False),
                    "elapsed_seconds": elapsed,
                }
            )
            print(f"[QUERY] {query_id}/{len(queries)} status={result.get('status')} evidence={len(result.get('evidence', []))} elapsed={elapsed:.3f}s")

        result_fields = list(evidence_rows[0].keys()) if evidence_rows else [
            "query_id", "query", "final_rank", "chunk_id", "doc_id", "source_title", "journal", "pub_year",
            "retrieval_sources", "vector_rank", "keyword_rank", "fusion_score", "relevance_score", "recency_score",
            "authority_score", "final_score", "text",
        ]
        summary_fields = list(summaries[0].keys())
        _write_csv(results_path, evidence_rows, result_fields)
        _write_csv(summary_path, summaries, summary_fields)
        report_path.write_text(make_report(args, summaries, len(evidence_rows)), encoding="utf-8")
        print(f"[DONE] queries={len(summaries)} evidence_rows={len(evidence_rows)}")
        print(f"[RESULTS] {results_path}")
        print(f"[SUMMARY] {summary_path}")
        print(f"[REPORT] {report_path}")
        print(f"[LOG] {log_path}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
