from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from medical_rag.retrieval.vector_store import (
    DEFAULT_METADATA_FIELDS,
    DEFAULT_MODEL_NAME,
    Tee,
    clean_metadata,
    dir_size_bytes,
    elapsed_rate,
    encode_passages,
    ensure_output_dirs,
    ensure_project_hf_cache,
    human_seconds,
    load_sentence_transformer,
    markdown_table,
    now_iso,
    parse_part_filter,
    project_root_from_script,
    resolve_device,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a persistent Chroma index from PMC chunk parquet shards.")
    parser.add_argument("--chunks_manifest", required=True)
    parser.add_argument("--chunks_dir", default="artifacts/datasets/chunks")
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--persist_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--max_parts", type=int, default=None)
    parser.add_argument("--part_filter", default=None, help='Comma/range part ids, e.g. "0,2-4".')
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shutdown_on_success", action="store_true")
    return parser.parse_args()


def read_manifest(manifest_path: Path, chunks_dir: Path, part_filter: set[int] | None, max_parts: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            part_id = int(row["part_id"])
            if part_filter is not None and part_id not in part_filter:
                continue
            part_file = Path(row["part_file"])
            if not part_file.exists():
                candidate = chunks_dir / part_file.name
                if candidate.exists():
                    part_file = candidate
            row["part_id"] = part_id
            row["part_file"] = str(part_file)
            row["chunk_count"] = int(row.get("chunk_count") or 0)
            rows.append(row)
            if max_parts is not None and len(rows) >= max_parts:
                break
    return rows


def expected_chunks_for_run(parts: list[dict[str, Any]], max_chunks: int | None) -> int:
    total = sum(int(row["chunk_count"]) for row in parts)
    if max_chunks is not None:
        return min(max_chunks, total)
    return total


def load_progress(progress_path: Path) -> dict[str, dict[str, Any]]:
    if not progress_path.exists():
        return {}
    with progress_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["part_file"]: row for row in csv.DictReader(handle)}


def write_progress(progress_path: Path, rows: list[dict[str, Any]]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["part_id", "part_file", "chunk_count", "status", "indexed_at", "start_chunk_id", "end_chunk_id"]
    with progress_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def existing_ids(collection, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    try:
        result = collection.get(ids=ids, include=[])
        return set(result.get("ids", []))
    except Exception:
        return set()


def dataframe_batches(df, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield start, df.iloc[start : start + batch_size]


def make_report(stats: dict[str, Any], part_rows: list[dict[str, Any]]) -> str:
    part_preview = part_rows[:12]
    return f"""# 向量化与索引构建报告（{stats['output_prefix']}）

## 1. 任务目标

本轮读取既有全文 chunk Parquet，不重新解析 XML、不重新切 chunk，使用 BAAI/bge-base-en-v1.5 生成归一化向量，并写入持久化 Chroma collection。

## 2. 输入 chunk 数据集

- Manifest: `{stats['source_manifest']}`
- Expected chunks: `{stats['total_chunks_expected']}`
- Selected parts: `{stats['selected_part_count']}`
- Max chunks: `{stats.get('max_chunks') or 'none'}`

## 3. 模型与向量配置

- Embedding model: `{stats['embedding_model']}`
- Embedding dimension: `{stats['embedding_dimension']}`
- Device: `{stats['device']}`
- normalize_embeddings: `true`
- Chroma metric: `cosine`
- 文档 embedding 不加 query instruction；查询脚本会自动添加 BGE query instruction。

## 4. Chroma Collection

- Collection: `{stats['collection_name']}`
- Persist dir: `{stats['persist_dir']}`
- Collection metadata: `{{"hnsw:space": "cosine"}}`

## 5. 写入规模与校验

- Total vectors indexed: `{stats['total_vectors_indexed']}`
- Collection count: `{stats['collection_count']}`
- Count matched expected: `{stats['count_matched_expected']}`
- Elapsed: `{human_seconds(float(stats['elapsed_seconds']))}`
- Index size: `{stats['index_size_mb']:.3f} MB` / `{stats['index_size_gb']:.3f} GB`

## 6. Chunk Token 统计

- mean: `{stats['chunk_size_stats']['mean']}`
- min: `{stats['chunk_size_stats']['min']}`
- max: `{stats['chunk_size_stats']['max']}`
- p95: `{stats['chunk_size_stats']['p95']}`

## 7. Metadata 字段

`{', '.join(stats['metadata_fields'])}`

## 8. Part 摘要

{markdown_table(part_preview, ['part_id', 'part_file', 'chunk_count', 'status', 'indexed_at'])}

## 9. 结论

- chroma_created: `{str(stats['chroma_created']).lower()}`
- embedding_created: `{str(stats['embedding_created']).lower()}`
- 后续验证脚本: `scripts/13_validate_chroma_index.py`
- 手动查询脚本: `scripts/14_query_chroma_index.py`
"""


def main() -> int:
    args = parse_args()
    project_root = project_root_from_script(__file__)
    ensure_output_dirs(project_root)
    ensure_project_hf_cache(project_root)

    log_path = project_root / "logs" / f"12_build_chroma_bge_index_{args.output_prefix}.log"
    tee = Tee(log_path)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee

    try:
        import chromadb
        import pandas as pd

        start_time = time.time()
        manifest_path = (project_root / args.chunks_manifest).resolve()
        chunks_dir = (project_root / args.chunks_dir).resolve()
        persist_dir = (project_root / args.persist_dir).resolve()
        progress_path = project_root / "artifacts/metrics/t008_vector_index" / f"vector_index_progress_{args.output_prefix}.csv"
        part_summary_path = project_root / "artifacts/metrics/t008_vector_index" / f"vector_index_part_summary_{args.output_prefix}.csv"
        stats_path = project_root / "artifacts/metrics/t008_vector_index" / f"vector_index_stats_{args.output_prefix}.json"
        report_path = project_root / "reports/formal" / f"向量化与索引构建报告_{args.output_prefix}.md"

        print(f"[START] {now_iso()}")
        print(f"[PROJECT_ROOT] {project_root}")
        print(f"[HF_HOME] {os.environ.get('HF_HOME')}")
        print(f"[MANIFEST] {manifest_path}")
        print(f"[PERSIST_DIR] {persist_dir}")

        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing chunks manifest: {manifest_path}")

        selected_parts = read_manifest(manifest_path, chunks_dir, parse_part_filter(args.part_filter), args.max_parts)
        if not selected_parts:
            raise ValueError("No parquet parts selected from manifest.")
        expected_total = expected_chunks_for_run(selected_parts, args.max_chunks)
        print(f"[PLAN] selected_parts={len(selected_parts)} expected_chunks={expected_total}")

        if args.force and persist_dir.exists():
            print(f"[FORCE] Removing existing persist dir: {persist_dir}")
            shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(persist_dir))
        if args.force:
            try:
                client.delete_collection(args.collection_name)
                print(f"[FORCE] Deleted existing collection: {args.collection_name}")
            except Exception:
                pass
        collection = client.get_or_create_collection(
            name=args.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        existing_count = collection.count()
        progress_map = load_progress(progress_path)
        if existing_count > 0 and not args.force and not progress_map:
            print(
                f"Collection {args.collection_name} already has {existing_count} vectors. "
                "No progress file was found, so the script will scan selected parts and skip existing ids."
            )

        device = resolve_device(args.device)
        print(f"[MODEL] Loading {args.model_name} on device={device}")
        model = load_sentence_transformer(args.model_name, device, project_root)
        embedding_dimension = int(model.get_sentence_embedding_dimension())
        print(f"[MODEL] embedding_dimension={embedding_dimension}")

        indexed = 0
        processed_chunks = 0
        token_counts: list[int] = []
        part_rows: list[dict[str, Any]] = []
        max_chunks_remaining = args.max_chunks

        for part in selected_parts:
            part_file = project_root / part["part_file"] if not Path(part["part_file"]).is_absolute() else Path(part["part_file"])
            progress_key = str(part_file)
            previous = progress_map.get(progress_key) or progress_map.get(str(part["part_file"]))
            if previous and previous.get("status") == "complete" and not args.force and args.max_chunks is None:
                print(f"[SKIP] completed part {part['part_id']}: {part_file}")
                part_rows.append(previous)
                indexed += int(previous.get("chunk_count") or 0)
                continue
            if max_chunks_remaining is not None and max_chunks_remaining <= 0:
                break
            if not part_file.exists():
                raise FileNotFoundError(f"Missing parquet part: {part_file}")

            print(f"[PART] part_id={part['part_id']} file={part_file}")
            df = pd.read_parquet(part_file)
            if max_chunks_remaining is not None and len(df) > max_chunks_remaining:
                df = df.iloc[:max_chunks_remaining].copy()
            required = {"chunk_id", "text"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{part_file} missing required columns: {sorted(missing)}")

            part_indexed = 0
            start_chunk_id = str(df.iloc[0]["chunk_id"]) if len(df) else ""
            end_chunk_id = str(df.iloc[-1]["chunk_id"]) if len(df) else ""
            if "token_count" in df.columns:
                token_counts.extend([int(x) for x in df["token_count"].fillna(0).tolist()])

            for batch_start, batch in dataframe_batches(df, args.batch_size):
                ids = [str(value) for value in batch["chunk_id"].tolist()]
                already = existing_ids(collection, ids) if not args.force else set()
                keep_positions = [idx for idx, chunk_id in enumerate(ids) if chunk_id not in already]
                if not keep_positions:
                    continue
                kept = batch.iloc[keep_positions]
                kept_ids = [ids[idx] for idx in keep_positions]
                docs = [str(text) for text in kept["text"].fillna("").tolist()]
                embeddings = encode_passages(model, docs, batch_size=args.batch_size)
                metas = [clean_metadata(row) for row in kept.to_dict(orient="records")]
                collection.add(ids=kept_ids, documents=docs, embeddings=embeddings, metadatas=metas)

                part_indexed += len(kept_ids)
                indexed += len(kept_ids)
                processed_chunks += len(kept_ids)
                rate, eta = elapsed_rate(indexed, expected_total, start_time)
                if part_indexed == len(kept_ids) or part_indexed % max(args.batch_size * 20, 1) == 0:
                    print(
                        f"[PROGRESS] indexed={indexed}/{expected_total} "
                        f"part={part['part_id']} batch_start={batch_start} rate={rate:.2f}/s eta={eta}"
                    )

            part_row = {
                "part_id": part["part_id"],
                "part_file": str(part_file),
                "chunk_count": len(df),
                "status": "complete",
                "indexed_at": now_iso(),
                "start_chunk_id": start_chunk_id,
                "end_chunk_id": end_chunk_id,
            }
            part_rows.append(part_row)
            write_progress(progress_path, part_rows)
            if max_chunks_remaining is not None:
                max_chunks_remaining -= len(df)

        collection_count = collection.count()
        elapsed = time.time() - start_time
        if token_counts:
            series = pd.Series(token_counts)
            chunk_stats = {
                "mean": round(float(series.mean()), 3),
                "max": int(series.max()),
                "min": int(series.min()),
                "p95": round(float(series.quantile(0.95)), 3),
            }
        else:
            chunk_stats = {"mean": 0, "max": 0, "min": 0, "p95": 0}
        size_bytes = dir_size_bytes(persist_dir)
        stats = {
            "output_prefix": args.output_prefix,
            "collection_name": args.collection_name,
            "persist_dir": str(persist_dir),
            "total_chunks_expected": expected_total,
            "total_vectors_indexed": collection_count,
            "vectors_added_this_run": indexed,
            "collection_count": collection_count,
            "count_matched_expected": collection_count == expected_total,
            "embedding_model": args.model_name,
            "embedding_dimension": embedding_dimension,
            "index_built_at": now_iso(),
            "device": device,
            "batch_size": args.batch_size,
            "source_manifest": str(manifest_path),
            "selected_part_count": len(selected_parts),
            "max_chunks": args.max_chunks,
            "chunk_size_stats": chunk_stats,
            "metadata_fields": DEFAULT_METADATA_FIELDS,
            "chroma_created": collection_count == expected_total,
            "embedding_created": collection_count == expected_total,
            "index_size_mb": round(size_bytes / 1024 / 1024, 3),
            "index_size_gb": round(size_bytes / 1024 / 1024 / 1024, 3),
            "elapsed_seconds": round(elapsed, 3),
        }

        part_summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(part_rows).to_csv(part_summary_path, index=False)
        write_json(stats_path, stats)
        report_path.write_text(make_report(stats, part_rows), encoding="utf-8")

        print(f"[DONE] collection_count={collection_count} expected={expected_total} elapsed={human_seconds(elapsed)}")
        print(f"[STATS] {stats_path}")
        print(f"[REPORT] {report_path}")
        if collection_count != expected_total:
            raise RuntimeError(f"Collection count mismatch: {collection_count} != {expected_total}")

        sys.stdout.flush()
        sys.stderr.flush()
        if args.shutdown_on_success:
            print("[SHUTDOWN] Success criteria met. Running sync then shutdown -h now.")
            sys.stdout.flush()
            subprocess.run(["sync"], check=False)
            subprocess.run(["shutdown", "-h", "now"], check=False)
        return 0
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
