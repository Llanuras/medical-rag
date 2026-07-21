from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_rag.retrieval.bm25 import DEFAULT_BM25_DIR, build_bm25_index
from medical_rag.retrieval.vector_store import Tee, human_seconds, now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a persistent BM25 index from PMC chunk parquet shards.")
    parser.add_argument(
        "--chunks_manifest",
        default="artifacts/datasets/chunks/pmc_chunks_limit153121_manifest.csv",
    )
    parser.add_argument("--chunks_dir", default="artifacts/datasets/chunks")
    parser.add_argument("--output_dir", default=str(DEFAULT_BM25_DIR))
    parser.add_argument("--output_prefix", default="limit153121")
    parser.add_argument("--max_parts", type=int, default=None)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def make_report(stats: dict) -> str:
    return f"""# BM25关键词索引构建报告（{stats['output_prefix']}）

## 任务边界

本轮直接读取既有 chunk Parquet，不重新解析 XML、不重新切 chunk、不生成 embedding、不重建 Chroma。

## 输入与范围

- Manifest: `{stats['source_manifest']}`
- Selected parts: `{stats['selected_part_ids']}`
- Max parts: `{stats['max_parts']}`
- Max chunks: `{stats['max_chunks']}`
- Indexed chunks: `{stats['document_count']}`

## BM25 配置

- Algorithm: `{stats['algorithm']}`
- Tokenizer: `{stats['tokenizer']}`
- k1: `{stats['k1']}`
- b: `{stats['b']}`
- Vocabulary size: `{stats['vocabulary_size']}`
- Average tokenized document length: `{stats['average_document_length']:.3f}`

## 产物

- Index: `{stats['index_path']}`
- Chunk store: `{stats['chunk_store_path']}`
- Build stats: `bm25_stats.json`
- Elapsed: `{human_seconds(float(stats['elapsed_seconds']))}`

## 说明

医学分词保留英文医学缩写、数字、连字号/斜杠复合词，并为中文查询保留短 n-gram。全量 Python 内存 BM25 的时间、内存与磁盘开销尚未在本报告中验证；局部验收不能替代全量结论。
"""


def main() -> int:
    args = parse_args()
    metrics_dir = PROJECT_ROOT / "artifacts/metrics/t014_multipath_retrieval"
    reports_dir = PROJECT_ROOT / "reports/formal"
    logs_dir = PROJECT_ROOT / "logs"
    for path in (metrics_dir, reports_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"18_build_bm25_index_{args.output_prefix}.log"
    tee = Tee(log_path)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        print(f"[START] {now_iso()}")
        stats = build_bm25_index(
            _resolve(args.chunks_manifest),
            _resolve(args.chunks_dir),
            _resolve(args.output_dir),
            output_prefix=args.output_prefix,
            max_parts=args.max_parts,
            max_chunks=args.max_chunks,
            force=args.force,
        )
        metrics_path = metrics_dir / f"bm25_stats_{args.output_prefix}.json"
        report_path = reports_dir / f"BM25关键词索引构建报告_{args.output_prefix}.md"
        metrics_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(make_report(stats), encoding="utf-8")
        print(f"[DONE] indexed={stats['document_count']} vocabulary={stats['vocabulary_size']}")
        print(f"[METRICS] {metrics_path}")
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
