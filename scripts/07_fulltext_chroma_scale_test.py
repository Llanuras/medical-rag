from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
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

from medical_rag.common.pmc import ensure_output_dirs, non_empty, read_records_csv, setup_tee, write_markdown

HASH_EMBED_DIM = 384
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?")

DEFAULT_QUERIES = [
    "Plasmodium falciparum gene expression malaria vaccine candidates",
    "cancer apoptosis p53 cell cycle",
    "HIV antiretroviral therapy resistance",
    "diabetes insulin glucose metabolism",
    "inflammation cytokines immune response",
]


class HashingEmbeddingFunction:
    """Small deterministic local embedding for Chroma scale tests.

    This is intentionally dependency-light and network-free. It is suitable for
    validating ingestion scale and keyword-style retrieval, not for final RAG
    semantic-quality evaluation.
    """

    def __init__(self, dim: int = HASH_EMBED_DIM):
        self.dim = dim

    def name(self) -> str:
        return f"local_hashing_bow_{self.dim}"

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def embed_documents(self, input):
        return self.__call__(input)

    def embed_query(self, input):
        if isinstance(input, str):
            return self._embed(input)
        return self.__call__(input)

    def _embed(self, text: str) -> list[float]:
        counts: Counter[int] = Counter()
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little", signed=False)
            index = value % self.dim
            sign = 1.0 if (value >> 63) == 0 else -1.0
            counts[index] += sign
        if not counts:
            return [0.0] * self.dim
        norm = math.sqrt(sum(v * v for v in counts.values()))
        vector = [0.0] * self.dim
        for index, value in counts.items():
            vector[index] = float(value / norm)
        return vector


class HashingTfidfEmbeddingFunction:
    def __init__(self, texts: list[str], dim: int = HASH_EMBED_DIM):
        self.dim = dim
        doc_freq: Counter[int] = Counter()
        for text in texts:
            indexes = {self._index(token) for token in TOKEN_RE.findall(text.lower())}
            doc_freq.update(indexes)
        doc_count = max(1, len(texts))
        self.idf = {
            index: math.log((1 + doc_count) / (1 + freq)) + 1.0
            for index, freq in doc_freq.items()
        }

    def name(self) -> str:
        return f"local_hashing_tfidf_{self.dim}"

    def embed_documents(self, input):
        return [self._embed(text) for text in input]

    def embed_query(self, input):
        if isinstance(input, str):
            return self._embed(input)
        return self.embed_documents(input)

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False) % self.dim

    def _embed(self, text: str) -> list[float]:
        counts: Counter[int] = Counter()
        for token in TOKEN_RE.findall(text.lower()):
            counts[self._index(token)] += 1
        if not counts:
            return [0.0] * self.dim
        vector = [0.0] * self.dim
        for index, count in counts.items():
            vector[index] = float((1 + math.log(count)) * self.idf.get(index, 1.0))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [float(value / norm) for value in vector]
        return vector


class TfidfEmbeddingFunction:
    def __init__(self, texts: list[str], max_features: int = HASH_EMBED_DIM):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_+-]{1,}\b",
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
            stop_words="english",
        )
        self.vectorizer.fit(texts)

    def name(self) -> str:
        return f"local_tfidf_{self.max_features}"

    def embed_documents(self, input):
        matrix = self.vectorizer.transform(input).astype("float32")
        return matrix.toarray().tolist()

    def embed_query(self, input):
        if isinstance(input, str):
            return self.embed_documents([input])[0]
        return self.embed_documents(input)


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            full = Path(root) / name
            try:
                total += full.stat().st_size
            except OSError:
                pass
    return total


def approx_token_len(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def load_token_splitter(chunk_size: int, chunk_overlap: int) -> tuple[RecursiveCharacterTextSplitter, str]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        return splitter, "huggingface_tokenizer_local:sentence-transformers/all-MiniLM-L6-v2"
    except Exception as exc:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=approx_token_len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        return splitter, f"regex_approx_token_count:fallback_after_{type(exc).__name__}"


def text_or_empty(node) -> str:
    if node is None:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def extract_body_sections(xml_path: Path) -> list[dict[str, str]]:
    raw = xml_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml-xml")
    body = soup.find("body")
    if body is None:
        return []
    top_sections = body.find_all("sec", recursive=False)
    sections = []
    for idx, sec in enumerate(top_sections):
        title_node = sec.find("title", recursive=False)
        section_title = text_or_empty(title_node) or f"Body section {idx + 1}"
        if title_node is not None:
            title_node.extract()
        section_text = text_or_empty(sec)
        if section_text:
            sections.append(
                {
                    "section_title": section_title,
                    "section_index": str(idx),
                    "section_text": section_text,
                    "section_source": "xml_top_level_sec",
                }
            )
    if not sections:
        body_text = text_or_empty(body)
        if body_text:
            sections.append(
                {
                    "section_title": "Body",
                    "section_index": "0",
                    "section_text": body_text,
                    "section_source": "xml_body_fallback",
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


def split_piece(text: str, splitter: RecursiveCharacterTextSplitter, no_split_under: int) -> list[str]:
    if no_split_under > 0 and approx_token_len(text) <= no_split_under:
        return [text]
    return splitter.split_text(text)


def chunk_record(
    row: pd.Series,
    data_dir: Path,
    splitter: RecursiveCharacterTextSplitter,
    strategy: str,
    no_split_under: int,
) -> list[dict]:
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
    chunks = []
    title = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    source_file = str(row.get("source_file", ""))
    xml_path = data_dir / source_file

    if strategy == "whole_document":
        text = str(row.get("text_full", "")).strip()
        if text:
            return [
                {
                    "text": text,
                    "meta": {
                        **base_meta,
                        "section_title": "Whole document",
                        "section_index": "0",
                        "section_source": "whole_document_no_split",
                        "within_section_chunk_index": 0,
                        "within_section_chunk_count": 1,
                    },
                }
            ]

    if strategy == "sliding_window":
        text = str(row.get("text_full", "")).strip()
        full_chunks = split_piece(text, splitter, no_split_under=0) if text else []
        for idx, chunk in enumerate(full_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "meta": {
                        **base_meta,
                        "section_title": "Full text sliding window",
                        "section_index": "0",
                        "section_source": "full_text_sliding_window",
                        "within_section_chunk_index": idx,
                        "within_section_chunk_count": len(full_chunks),
                    },
                }
            )
    else:
        front_text = "\n\n".join(part for part in [title, abstract] if part)
        if front_text:
            front_chunks = split_piece(front_text, splitter, no_split_under)
            for idx, chunk in enumerate(front_chunks):
                chunks.append(
                    {
                        "text": chunk,
                        "meta": {
                            **base_meta,
                            "section_title": "Title and abstract",
                            "section_index": "-1",
                            "section_source": "record_title_abstract",
                            "within_section_chunk_index": idx,
                            "within_section_chunk_count": len(front_chunks),
                        },
                    }
                )

        sections = extract_body_sections(xml_path) if xml_path.exists() else []
        if not sections:
            body = str(row.get("body", "")).strip()
            if body:
                sections = [
                    {
                        "section_title": "Body",
                        "section_index": "0",
                        "section_text": body,
                        "section_source": "csv_body_fallback",
                    }
                ]

        for section in sections:
            section_title = section["section_title"]
            section_text = section["section_text"]
            text = f"{section_title}\n\n{section_text}" if section_title else section_text
            section_chunks = split_piece(text, splitter, no_split_under)
            for idx, chunk in enumerate(section_chunks):
                chunks.append(
                    {
                        "text": chunk,
                        "meta": {
                            **base_meta,
                            "section_title": section_title,
                            "section_index": section["section_index"],
                            "section_source": section["section_source"],
                            "within_section_chunk_index": idx,
                            "within_section_chunk_count": len(section_chunks),
                        },
                    }
                )

    total = len(chunks)
    for idx, item in enumerate(chunks):
        item["meta"]["chunk_index"] = idx
        item["meta"]["chunk_count"] = total
        item["meta"]["chunk_chars"] = len(item["text"])
        item["meta"]["chunk_token_len_approx"] = approx_token_len(item["text"])
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


def strategy_comparison_rows(
    eligible: pd.DataFrame,
    data_dir: Path,
    splitter: RecursiveCharacterTextSplitter,
    no_split_under: int,
) -> list[dict]:
    rows = []
    for strategy in ["whole_document", "sliding_window", "semantic_section_recursive"]:
        started = time.perf_counter()
        per_record = []
        section_sources: Counter[str] = Counter()
        section_chunk_counts = []
        for _, row in eligible.iterrows():
            record_chunks = chunk_record(row, data_dir, splitter, strategy, no_split_under)
            per_record.append(len(record_chunks))
            for item in record_chunks:
                section_sources[item["meta"].get("section_source", "")] += 1
                section_chunk_counts.append(int(item["meta"].get("chunk_token_len_approx", 0)))
        counts = pd.Series(per_record, dtype="int64")
        token_lens = pd.Series(section_chunk_counts, dtype="int64")
        rows.append(
            {
                "strategy": strategy,
                "records": len(eligible),
                "total_chunks": int(counts.sum()) if len(counts) else 0,
                "chunks_per_record_mean": float(counts.mean()) if len(counts) else 0,
                "chunks_per_record_median": float(counts.median()) if len(counts) else 0,
                "chunks_per_record_p95": float(counts.quantile(0.95)) if len(counts) else 0,
                "chunk_token_len_approx_mean": float(token_lens.mean()) if len(token_lens) else 0,
                "chunk_token_len_approx_p95": float(token_lens.quantile(0.95)) if len(token_lens) else 0,
                "seconds": f"{time.perf_counter() - started:.3f}",
                "section_source_breakdown": "; ".join(f"{k}={v}" for k, v in sorted(section_sources.items())),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/datasets/records/pmc_records_limit3028.csv")
    parser.add_argument("--quality", default="")
    parser.add_argument("--tokens", default="artifacts/metrics/t002_corpus_analysis/token_length_records_limit3028.csv")
    parser.add_argument("--data_dir", default="data/raw/pmc_oa_comm")
    parser.add_argument("--output_prefix", default="limit3028")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--persist_dir", default="archive/experiments/indexes/chroma_fulltext_limit3028_hash")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--chunk_overlap", type=int, default=80)
    parser.add_argument("--embedding_backend", choices=["hash", "hash_tfidf", "tfidf"], default="hash_tfidf")
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument(
        "--strategy",
        choices=["semantic_section_recursive", "sliding_window", "whole_document"],
        default="semantic_section_recursive",
    )
    parser.add_argument(
        "--no_split_under",
        type=int,
        default=400,
        help="For semantic_section_recursive, keep title/abstract or section whole when approx tokens are <= this value.",
    )
    parser.add_argument("--compare_strategies", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    ensure_output_dirs()
    log_path = Path(f"logs/07_fulltext_chroma_scale_test_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        started = time.perf_counter()
        records = read_records_csv(Path(args.input))
        if args.limit > 0:
            records = records.head(args.limit).copy()
        records["text_full"] = records["text_full"].fillna("")
        eligible = records[non_empty(records["text_full"])].copy()
        quality_filter = "not_applied"
        if args.quality:
            quality = pd.read_csv(args.quality, dtype=str).fillna("")
            eligible = eligible.merge(quality[["record_id", "quality_decision"]], on="record_id", how="left")
            eligible = eligible[eligible["quality_decision"].isin(["keep", "keep_with_warning"])].copy()
            quality_filter = "keep_or_keep_with_warning"

        splitter, splitter_source = load_token_splitter(args.chunk_size, args.chunk_overlap)
        data_dir = Path(args.data_dir)
        print(f"Input records: {len(records)}")
        print(f"Eligible full-text records: {len(eligible)}")
        print(f"Strategy: {args.strategy}")
        print(f"Chunk config: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}, no_split_under={args.no_split_under}")
        print(f"Embedding backend: {args.embedding_backend}, dim={args.embedding_dim}")
        print(f"Splitter: {splitter_source}")

        comparison_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_chunk_strategy_comparison_{args.output_prefix}.csv")
        comparison_note = "not requested"
        if args.compare_strategies:
            comparison_rows = strategy_comparison_rows(eligible, data_dir, splitter, args.no_split_under)
            pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
            comparison_note = str(comparison_path)
            print("Wrote", comparison_path)

        chunk_started = time.perf_counter()
        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []
        manifest_rows: list[dict] = []
        for record_pos, (_, row) in enumerate(eligible.iterrows(), start=1):
            record_chunks = chunk_record(row, data_dir, splitter, args.strategy, args.no_split_under)
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
                        "chunk_token_len_approx": meta["chunk_token_len_approx"],
                    }
                )
            if record_pos % 250 == 0:
                print(f"Chunked {record_pos}/{len(eligible)} records; chunks so far: {len(docs)}")
        chunk_seconds = time.perf_counter() - chunk_started
        print(f"Chunking seconds: {chunk_seconds:.2f}")
        print(f"Total chunks: {len(docs)}")

        persist_dir = Path(args.persist_dir)
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        embedding_started = time.perf_counter()
        if args.embedding_backend == "tfidf":
            embedding = TfidfEmbeddingFunction(docs, max_features=args.embedding_dim)
        elif args.embedding_backend == "hash_tfidf":
            embedding = HashingTfidfEmbeddingFunction(docs, dim=args.embedding_dim)
        else:
            embedding = HashingEmbeddingFunction(dim=args.embedding_dim)
        embedding_seconds = time.perf_counter() - embedding_started
        print(f"Embedding setup seconds: {embedding_seconds:.2f}")

        add_started = time.perf_counter()
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name="pmc_fulltext_scale",
            metadata={
                "description": f"3028 PMC full-text chunk scale test with local {embedding.name()} embeddings",
                "hnsw:space": "cosine",
            },
        )
        for batch_no, batch_indexes in enumerate(batched(list(range(len(docs))), args.batch_size), start=1):
            batch_docs = [docs[i] for i in batch_indexes]
            collection.add(
                ids=[ids[i] for i in batch_indexes],
                documents=batch_docs,
                metadatas=[metadatas[i] for i in batch_indexes],
                embeddings=embedding.embed_documents(batch_docs),
            )
            print(f"Added batch {batch_no}; total added: {min(batch_no * args.batch_size, len(docs))}/{len(docs)}")
        add_seconds = time.perf_counter() - add_started
        count_after_add = collection.count()
        print(f"Chroma add seconds: {add_seconds:.2f}")
        print(f"Chroma collection count: {count_after_add}")

        query_rows: list[dict] = []
        queries = args.query or DEFAULT_QUERIES
        query_started = time.perf_counter()
        for query in queries:
            result = collection.query(query_embeddings=[embedding.embed_query(query)], n_results=args.top_k)
            result_ids = result.get("ids", [[]])[0]
            result_docs = result.get("documents", [[]])[0]
            result_metas = result.get("metadatas", [[]])[0]
            result_distances = result.get("distances", [[]])[0]
            for rank, (doc_id, text, meta, distance) in enumerate(
                zip(result_ids, result_docs, result_metas, result_distances),
                start=1,
            ):
                snippet = " ".join((text or "").split())[:320]
                query_rows.append(
                    {
                        "query": query,
                        "rank": rank,
                        "distance": distance,
                        "id": doc_id,
                        "record_id": meta.get("record_id", ""),
                        "title": meta.get("title", ""),
                        "section_title": meta.get("section_title", ""),
                        "source_file": meta.get("source_file", ""),
                        "snippet": snippet,
                    }
                )
        query_seconds = time.perf_counter() - query_started

        db_bytes = dir_size_bytes(persist_dir)
        total_seconds = time.perf_counter() - started
        chunk_counts = pd.Series([row["chunk_count"] for row in manifest_rows], dtype="int64")
        per_record_counts = pd.DataFrame(manifest_rows).groupby("record_id").size() if manifest_rows else pd.Series([], dtype="int64")
        section_sources = Counter(row["section_source"] for row in manifest_rows)
        summary_rows = [
            {"metric": "input_records", "value": len(records)},
            {"metric": "eligible_fulltext_records", "value": len(eligible)},
            {"metric": "quality_filter", "value": quality_filter},
            {"metric": "strategy", "value": args.strategy},
            {"metric": "chunk_size", "value": args.chunk_size},
            {"metric": "chunk_overlap", "value": args.chunk_overlap},
            {"metric": "no_split_under", "value": args.no_split_under},
            {"metric": "splitter_source", "value": splitter_source},
            {"metric": "embedding_backend", "value": embedding.name()},
            {"metric": "embedding_setup_seconds", "value": f"{embedding_seconds:.3f}"},
            {"metric": "strategy_comparison_path", "value": comparison_note},
            {"metric": "total_chunks", "value": len(docs)},
            {"metric": "chroma_collection_count", "value": count_after_add},
            {"metric": "chunks_per_record_mean", "value": float(per_record_counts.mean()) if len(per_record_counts) else 0},
            {"metric": "chunks_per_record_median", "value": float(per_record_counts.median()) if len(per_record_counts) else 0},
            {"metric": "chunks_per_record_p95", "value": float(per_record_counts.quantile(0.95)) if len(per_record_counts) else 0},
            {"metric": "chunk_token_len_approx_mean", "value": float(pd.Series([row["chunk_token_len_approx"] for row in manifest_rows]).mean()) if manifest_rows else 0},
            {"metric": "chunking_seconds", "value": f"{chunk_seconds:.3f}"},
            {"metric": "chroma_add_seconds", "value": f"{add_seconds:.3f}"},
            {"metric": "query_seconds_total", "value": f"{query_seconds:.3f}"},
            {"metric": "total_seconds", "value": f"{total_seconds:.3f}"},
            {"metric": "persist_directory", "value": str(persist_dir)},
            {"metric": "persist_size_bytes", "value": db_bytes},
            {"metric": "persist_size_mb", "value": f"{db_bytes / 1024**2:.2f}"},
            {"metric": "batch_size", "value": args.batch_size},
            {"metric": "top_k", "value": args.top_k},
        ]
        for source, count in sorted(section_sources.items()):
            summary_rows.append({"metric": f"chunks_from_{source}", "value": count})

        summary_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_chroma_scale_summary_{args.output_prefix}.csv")
        query_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_chroma_query_results_{args.output_prefix}.csv")
        manifest_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_chroma_document_manifest_{args.output_prefix}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        write_csv(query_path, query_rows)
        write_csv(manifest_path, manifest_rows)

        first_results = "\n".join(
            f"- `{row['query']}` -> rank {row['rank']} `{row['record_id']}` / {row['section_title']}: {row['snippet']}"
            for row in query_rows
            if row["rank"] == 1
        )
        body = f"""
## 实验目标

本实验在本地 `3028` 篇 PMC 全文上执行全文 chunk + Chroma 持久化写入，记录实际 chunks 数、耗时、库大小和检索样例，用于判断全文 RAG 下一阶段建库是否可行。

## 切分与入库策略

- 输入：`{args.input}`
- 入库记录：`text_full` 非空记录；质量过滤：`{quality_filter}`
- 切分策略：`{args.strategy}`。默认口径为章节优先；每篇保留 `Title and abstract`，正文优先按 XML 顶层 `sec` 保留 `section_title`，短章节整体保留，长章节再做递归切分
- chunk 参数：`chunk_size={args.chunk_size}`, `chunk_overlap={args.chunk_overlap}`, `no_split_under={args.no_split_under}`
- splitter：`{splitter_source}`
- embedding：`{embedding.name()}`，用于本地规模验证和关键词型检索，不代表最终 MiniLM/生产 embedding 的语义效果
- Chroma 目录：`{persist_dir}`
- 三种切分策略对比：`{comparison_note}`

## 实测结果

- 输入记录数：`{len(records)}`
- 实际入库全文记录数：`{len(eligible)}`
- 实际写入 chunks：`{len(docs)}`
- Chroma collection count：`{count_after_add}`
- chunks/篇 mean：`{(float(per_record_counts.mean()) if len(per_record_counts) else 0):.2f}`
- chunks/篇 median：`{(float(per_record_counts.median()) if len(per_record_counts) else 0):.2f}`
- chunks/篇 p95：`{(float(per_record_counts.quantile(0.95)) if len(per_record_counts) else 0):.2f}`
- embedding 准备耗时：`{embedding_seconds:.2f}` 秒
- 切分耗时：`{chunk_seconds:.2f}` 秒
- Chroma 写入耗时：`{add_seconds:.2f}` 秒
- 查询总耗时：`{query_seconds:.2f}` 秒
- 总耗时：`{total_seconds:.2f}` 秒
- 向量库大小：`{db_bytes / 1024**2:.2f}` MB

## 检索 sanity check

{first_results}

完整 top-{args.top_k} 检索结果见 `{query_path}`。

## 初步判断

本机可以完成 `3028` 篇全文 chunk + Chroma 持久化写入。该规模下的主要增长来自全文切分后的 chunk 数，而不是原始文献篇数。当前本地 embedding 能验证库构建、metadata 溯源和关键词检索链路；后续若要评估最终 RAG 答案质量，应将 embedding backend 替换为正式医学/通用 embedding 模型，并复用同一 manifest 与检索评估表结构。
"""
        doc_path = Path(f"reports/technical/07_fulltext_chroma_scale_test_notes_{args.output_prefix}.md")
        write_markdown(doc_path, "3028 篇全文 Chroma 规模验证说明", body)

        print("Wrote", summary_path)
        print("Wrote", query_path)
        print("Wrote", manifest_path)
        print("Wrote", doc_path)


if __name__ == "__main__":
    main()
