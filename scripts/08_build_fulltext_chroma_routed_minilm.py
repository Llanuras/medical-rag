from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import chromadb
import pandas as pd
from bs4 import BeautifulSoup
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from medical_rag.common.pmc import ensure_output_dirs, read_records_csv, setup_tee, write_markdown

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
WHOLE_DOC_TOKEN_LIMIT = 512
DEFAULT_QUERIES = [
    "Plasmodium falciparum intraerythrocytic developmental cycle transcriptome",
    "pRb inactivation mammary cells tumor initiation progression",
    "type 2 diabetes high protein diet insulin concentration",
    "immune response B cells master regulator",
    "SARS coronavirus spike protein trafficking",
]


def ensure_project_hf_cache(project_dir: Path) -> None:
    hf_home = project_dir / "artifacts/models/huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def token_len(tokenizer, text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def text_or_empty(node) -> str:
    if node is None:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def extract_top_level_sections(xml_path: Path) -> list[dict[str, str]]:
    raw = xml_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml-xml")
    body = soup.find("body")
    if body is None:
        return []
    sections = []
    for idx, sec in enumerate(body.find_all("sec", recursive=False)):
        title_node = sec.find("title", recursive=False)
        title = text_or_empty(title_node) or f"Body section {idx + 1}"
        if title_node is not None:
            title_node.extract()
        section_text = text_or_empty(sec)
        if section_text:
            sections.append(
                {
                    "section_index": str(idx),
                    "section_title": title,
                    "section_text": section_text,
                    "section_source": "xml_top_level_sec",
                }
            )
    return sections


def clean_metadata(meta: dict) -> dict:
    cleaned = {}
    for key, value in meta.items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def split_with_token_limit(
    text: str,
    splitter: RecursiveCharacterTextSplitter,
    tokenizer,
    keep_under: int = WHOLE_DOC_TOKEN_LIMIT,
) -> tuple[list[str], str]:
    if token_len(tokenizer, text) <= keep_under:
        return [text], "section_whole_under_512"
    return splitter.split_text(text), "section_recursive_over_512"


def route_name(row: pd.Series) -> str:
    full_len = int(row["full_token_len"])
    has_section = bool(row["has_any_section_title"])
    if full_len <= WHOLE_DOC_TOKEN_LIMIT:
        return "whole_document_under_512"
    if has_section:
        return "semantic_section"
    return "recursive_fallback_no_section"


def build_chunks_for_record(
    row: pd.Series,
    data_dir: Path,
    splitter: RecursiveCharacterTextSplitter,
    tokenizer,
) -> list[dict]:
    route = route_name(row)
    base_meta = {
        key: row.get(key, "")
        for key in [
            "record_id",
            "source_file",
            "title",
            "journal",
            "pub_year",
            "pmid",
            "pmcid",
            "article_type",
        ]
    }
    source_file = str(row.get("source_file", ""))
    full_text = str(row.get("text_full", "")).strip()
    chunks: list[dict] = []

    if route == "whole_document_under_512":
        chunks.append(
            {
                "text": full_text,
                "meta": {
                    **base_meta,
                    "route": route,
                    "chunk_strategy": "whole_document",
                    "section_index": "0",
                    "section_title": "Whole document",
                    "section_source": "whole_document_under_512",
                    "within_section_chunk_index": 0,
                    "within_section_chunk_count": 1,
                },
            }
        )
    elif route == "semantic_section":
        front_text = "\n\n".join(
            part for part in [str(row.get("title", "")).strip(), str(row.get("abstract", "")).strip()] if part
        )
        if front_text:
            front_chunks, front_strategy = split_with_token_limit(front_text, splitter, tokenizer)
            for idx, chunk in enumerate(front_chunks):
                chunks.append(
                    {
                        "text": chunk,
                        "meta": {
                            **base_meta,
                            "route": route,
                            "chunk_strategy": front_strategy,
                            "section_index": "-1",
                            "section_title": "Title and abstract",
                            "section_source": "record_title_abstract",
                            "within_section_chunk_index": idx,
                            "within_section_chunk_count": len(front_chunks),
                        },
                    }
                )
        xml_path = data_dir / source_file
        sections = extract_top_level_sections(xml_path) if xml_path.exists() else []
        for section in sections:
            section_text = f"{section['section_title']}\n\n{section['section_text']}"
            section_chunks, section_strategy = split_with_token_limit(section_text, splitter, tokenizer)
            for idx, chunk in enumerate(section_chunks):
                chunks.append(
                    {
                        "text": chunk,
                        "meta": {
                            **base_meta,
                            "route": route,
                            "chunk_strategy": section_strategy,
                            "section_index": section["section_index"],
                            "section_title": section["section_title"],
                            "section_source": section["section_source"],
                            "within_section_chunk_index": idx,
                            "within_section_chunk_count": len(section_chunks),
                        },
                    }
                )
    else:
        fallback_chunks = splitter.split_text(full_text)
        for idx, chunk in enumerate(fallback_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "meta": {
                        **base_meta,
                        "route": route,
                        "chunk_strategy": "recursive_fulltext_no_section",
                        "section_index": "0",
                        "section_title": "Full text fallback",
                        "section_source": "no_section_recursive_fallback",
                        "within_section_chunk_index": idx,
                        "within_section_chunk_count": len(fallback_chunks),
                    },
                }
            )

    total = len(chunks)
    for idx, item in enumerate(chunks):
        item["meta"]["chunk_index"] = idx
        item["meta"]["chunk_count"] = total
        item["meta"]["chunk_chars"] = len(item["text"])
        item["meta"]["chunk_token_len"] = token_len(tokenizer, item["text"])
    return chunks


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="artifacts/datasets/records/pmc_records_limit3028.csv")
    parser.add_argument("--sections", default="artifacts/metrics/t002_corpus_analysis/full_text_section_analysis_limit3028.csv")
    parser.add_argument("--data_dir", default="data/raw/pmc_oa_comm")
    parser.add_argument("--persist_dir", default="archive/experiments/indexes/chroma_fulltext_limit3028_routed_minilm")
    parser.add_argument("--output_prefix", default="limit3028_routed_minilm")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    project_dir = Path.cwd()
    ensure_project_hf_cache(project_dir)
    ensure_output_dirs()
    log_path = Path(f"logs/08_fulltext_chroma_routed_minilm_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        started = time.perf_counter()
        print("HF_HOME:", os.environ.get("HF_HOME"))
        print("Embedding model:", MODEL)

        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

        records = read_records_csv(Path(args.records))
        if args.limit > 0:
            records = records.head(args.limit).copy()
        sections = pd.read_csv(args.sections)
        df = records.merge(
            sections[
                [
                    "record_id",
                    "full_token_len",
                    "estimated_chunks_full",
                    "section_title_count",
                    "has_any_section_title",
                ]
            ],
            on="record_id",
            how="left",
        )
        df["full_token_len"] = pd.to_numeric(df["full_token_len"], errors="coerce").fillna(0).astype(int)
        df["has_any_section_title"] = df["has_any_section_title"].apply(parse_bool)
        df["route"] = df.apply(route_name, axis=1)
        route_counts = df["route"].value_counts().to_dict()
        print("Route counts:", route_counts)

        chunk_started = time.perf_counter()
        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []
        manifest_rows: list[dict] = []
        data_dir = Path(args.data_dir)
        for pos, (_, row) in enumerate(df.iterrows(), start=1):
            record_chunks = build_chunks_for_record(row, data_dir, splitter, tokenizer)
            for item in record_chunks:
                meta = clean_metadata(item["meta"])
                chunk_id = f"{meta['record_id']}::chunk_{int(meta['chunk_index']):05d}"
                ids.append(chunk_id)
                docs.append(item["text"])
                metadatas.append(meta)
                manifest_rows.append(
                    {
                        "id": chunk_id,
                        "record_id": meta["record_id"],
                        "route": meta["route"],
                        "chunk_strategy": meta["chunk_strategy"],
                        "source_file": meta["source_file"],
                        "title": meta["title"],
                        "section_title": meta["section_title"],
                        "section_index": meta["section_index"],
                        "section_source": meta["section_source"],
                        "chunk_index": meta["chunk_index"],
                        "chunk_count": meta["chunk_count"],
                        "within_section_chunk_index": meta["within_section_chunk_index"],
                        "within_section_chunk_count": meta["within_section_chunk_count"],
                        "chunk_chars": meta["chunk_chars"],
                        "chunk_token_len": meta["chunk_token_len"],
                    }
                )
            if pos % 250 == 0:
                print(f"Chunked {pos}/{len(df)} records; chunks so far: {len(docs)}")
        chunk_seconds = time.perf_counter() - chunk_started
        print(f"Chunking seconds: {chunk_seconds:.2f}")
        print(f"Total chunks: {len(docs)}")

        persist_dir = Path(args.persist_dir)
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        model_started = time.perf_counter()
        model = SentenceTransformer(MODEL)
        model_seconds = time.perf_counter() - model_started
        print(f"Model load seconds: {model_seconds:.2f}")

        add_started = time.perf_counter()
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name="pmc_fulltext_routed_minilm",
            metadata={
                "description": "3028 PMC full text routed chunks with all-MiniLM-L6-v2 embeddings",
                "hnsw:space": "cosine",
            },
        )
        for batch_no, batch_indexes in enumerate(batched(list(range(len(docs))), args.batch_size), start=1):
            batch_docs = [docs[i] for i in batch_indexes]
            embeddings = model.encode(
                batch_docs,
                batch_size=args.encode_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            collection.add(
                ids=[ids[i] for i in batch_indexes],
                documents=batch_docs,
                metadatas=[metadatas[i] for i in batch_indexes],
                embeddings=embeddings.tolist(),
            )
            added = min(batch_no * args.batch_size, len(docs))
            print(f"Added batch {batch_no}; total added: {added}/{len(docs)}")
        add_seconds = time.perf_counter() - add_started
        collection_count = collection.count()
        print(f"Chroma add seconds: {add_seconds:.2f}")
        print(f"Chroma collection count: {collection_count}")

        query_rows: list[dict] = []
        query_started = time.perf_counter()
        queries = args.query or DEFAULT_QUERIES
        query_embeddings = model.encode(
            queries,
            batch_size=len(queries),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for query, query_embedding in zip(queries, query_embeddings):
            result = collection.query(query_embeddings=[query_embedding.tolist()], n_results=args.top_k)
            for rank, (doc_id, text, meta, distance) in enumerate(
                zip(
                    result.get("ids", [[]])[0],
                    result.get("documents", [[]])[0],
                    result.get("metadatas", [[]])[0],
                    result.get("distances", [[]])[0],
                ),
                start=1,
            ):
                query_rows.append(
                    {
                        "query": query,
                        "rank": rank,
                        "distance": distance,
                        "id": doc_id,
                        "record_id": meta.get("record_id", ""),
                        "route": meta.get("route", ""),
                        "chunk_strategy": meta.get("chunk_strategy", ""),
                        "title": meta.get("title", ""),
                        "section_title": meta.get("section_title", ""),
                        "source_file": meta.get("source_file", ""),
                        "snippet": " ".join((text or "").split())[:320],
                    }
                )
        query_seconds = time.perf_counter() - query_started

        persist_bytes = dir_size_bytes(persist_dir)
        total_seconds = time.perf_counter() - started
        manifest_df = pd.DataFrame(manifest_rows)
        per_record_counts = manifest_df.groupby("record_id").size()
        route_chunk_counts = manifest_df["route"].value_counts().to_dict()
        strategy_chunk_counts = manifest_df["chunk_strategy"].value_counts().to_dict()
        source_chunk_counts = manifest_df["section_source"].value_counts().to_dict()
        chunk_token_lens = manifest_df["chunk_token_len"].astype(int)

        summary_rows = [
            {"metric": "input_records", "value": len(df)},
            {"metric": "embedding_model", "value": MODEL},
            {"metric": "hf_home", "value": os.environ.get("HF_HOME", "")},
            {"metric": "whole_document_under_512_records", "value": int(route_counts.get("whole_document_under_512", 0))},
            {"metric": "semantic_section_records", "value": int(route_counts.get("semantic_section", 0))},
            {"metric": "recursive_fallback_no_section_records", "value": int(route_counts.get("recursive_fallback_no_section", 0))},
            {"metric": "chunk_size", "value": CHUNK_SIZE},
            {"metric": "chunk_overlap", "value": CHUNK_OVERLAP},
            {"metric": "whole_doc_token_limit", "value": WHOLE_DOC_TOKEN_LIMIT},
            {"metric": "total_chunks", "value": len(docs)},
            {"metric": "chroma_collection_count", "value": collection_count},
            {"metric": "chunks_per_record_mean", "value": float(per_record_counts.mean())},
            {"metric": "chunks_per_record_median", "value": float(per_record_counts.median())},
            {"metric": "chunks_per_record_p95", "value": float(per_record_counts.quantile(0.95))},
            {"metric": "chunk_token_len_mean", "value": float(chunk_token_lens.mean())},
            {"metric": "chunk_token_len_p95", "value": float(chunk_token_lens.quantile(0.95))},
            {"metric": "chunking_seconds", "value": f"{chunk_seconds:.3f}"},
            {"metric": "model_load_seconds", "value": f"{model_seconds:.3f}"},
            {"metric": "chroma_add_seconds", "value": f"{add_seconds:.3f}"},
            {"metric": "query_seconds_total", "value": f"{query_seconds:.3f}"},
            {"metric": "total_seconds", "value": f"{total_seconds:.3f}"},
            {"metric": "persist_directory", "value": str(persist_dir)},
            {"metric": "persist_size_bytes", "value": persist_bytes},
            {"metric": "persist_size_mb", "value": f"{persist_bytes / 1024**2:.2f}"},
            {"metric": "batch_size", "value": args.batch_size},
            {"metric": "encode_batch_size", "value": args.encode_batch_size},
            {"metric": "top_k", "value": args.top_k},
        ]
        for route, count in sorted(route_chunk_counts.items()):
            summary_rows.append({"metric": f"chunks_route_{route}", "value": int(count)})
        for strategy, count in sorted(strategy_chunk_counts.items()):
            summary_rows.append({"metric": f"chunks_strategy_{strategy}", "value": int(count)})
        for source, count in sorted(source_chunk_counts.items()):
            summary_rows.append({"metric": f"chunks_source_{source}", "value": int(count)})

        summary_path = Path(f"artifacts/metrics/t005_routed_minilm/fulltext_chroma_routed_minilm_summary_{args.output_prefix}.csv")
        manifest_path = Path(f"artifacts/metrics/t005_routed_minilm/fulltext_chroma_routed_minilm_manifest_{args.output_prefix}.csv")
        query_path = Path(f"artifacts/metrics/t005_routed_minilm/fulltext_chroma_routed_minilm_query_results_{args.output_prefix}.csv")
        route_path = Path(f"artifacts/metrics/t005_routed_minilm/fulltext_chroma_routed_minilm_route_summary_{args.output_prefix}.csv")

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        manifest_df.to_csv(manifest_path, index=False)
        write_csv(query_path, query_rows)
        route_summary = []
        for route, group in manifest_df.groupby("route"):
            counts = group.groupby("record_id").size()
            route_summary.append(
                {
                    "route": route,
                    "records": int(df[df["route"] == route].shape[0]),
                    "chunks": int(group.shape[0]),
                    "chunks_per_record_mean": float(counts.mean()),
                    "chunks_per_record_median": float(counts.median()),
                    "chunks_per_record_p95": float(counts.quantile(0.95)),
                }
            )
        pd.DataFrame(route_summary).to_csv(route_path, index=False)

        rank1_lines = "\n".join(
            f"- `{row['query']}` -> `{row['record_id']}` / {row['section_title']}: {row['snippet']}"
            for row in query_rows
            if row["rank"] == 1
        )
        body = f"""
## 实验口径

本次为正式 3028 篇全文 chunk + Chroma 规模验证，严格按照既有全文结构分析结果做三路由：

- `whole_document_under_512`：`{int(route_counts.get('whole_document_under_512', 0))}` 篇，全文不分割。
- `semantic_section`：`{int(route_counts.get('semantic_section', 0))}` 篇，按 XML 顶层章节语义切分，并保留 `section_title` metadata；超长章节只在章节内部递归切分。
- `recursive_fallback_no_section`：`{int(route_counts.get('recursive_fallback_no_section', 0))}` 篇，无章节且全文超过 512 tokens，使用 `RecursiveCharacterTextSplitter` 兜底。

Embedding 使用项目缓存中的 `{MODEL}`，`HF_HOME={os.environ.get('HF_HOME', '')}`。

## 实测结果

- 实际写入 chunks：`{len(docs)}`
- Chroma collection count：`{collection_count}`
- chunks/篇 mean：`{float(per_record_counts.mean()):.2f}`
- chunks/篇 median：`{float(per_record_counts.median()):.2f}`
- chunks/篇 p95：`{float(per_record_counts.quantile(0.95)):.2f}`
- chunk token p95：`{float(chunk_token_lens.quantile(0.95)):.2f}`
- 切分耗时：`{chunk_seconds:.2f}` 秒
- 模型加载耗时：`{model_seconds:.2f}` 秒
- Chroma 写入耗时：`{add_seconds:.2f}` 秒
- 查询耗时：`{query_seconds:.2f}` 秒
- 总耗时：`{total_seconds:.2f}` 秒
- 向量库大小：`{persist_bytes / 1024**2:.2f}` MB
- Chroma 目录：`{persist_dir}`

## 检索 sanity check

{rank1_lines}

完整 top-{args.top_k} 结果见 `{query_path}`。

## 关于旧 07 报告

`reports/technical` 中多个 `07_fulltext_chroma_scale_test_*` 文件是此前错误口径和冒烟实验生成的中间文件，包含 hash/TF-IDF embedding 和非严格三路由切分，不应作为本阶段正式结论。正式结论以本报告和 `artifacts/metrics/t005_routed_minilm/fulltext_chroma_routed_minilm_*_{args.output_prefix}.csv` 为准。
"""
        doc_path = Path(f"reports/technical/08_fulltext_chroma_routed_minilm_report_{args.output_prefix}.md")
        write_markdown(doc_path, "3028 篇全文 routed chunk + all-MiniLM Chroma 规模验证", body)

        print("Wrote", summary_path)
        print("Wrote", route_path)
        print("Wrote", manifest_path)
        print("Wrote", query_path)
        print("Wrote", doc_path)


if __name__ == "__main__":
    main()
