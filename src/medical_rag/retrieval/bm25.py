from __future__ import annotations

import csv
import json
import math
import pickle
import re
import time
import unicodedata
from array import array
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from medical_rag.retrieval.vector_store import DEFAULT_METADATA_FIELDS, clean_metadata_value


DEFAULT_BM25_DIR = Path("artifacts/indexes/bm25/pmc_fulltext_bm25_limit153121")
DEFAULT_BM25_PART000_DIR = Path("artifacts/indexes/bm25/pmc_fulltext_bm25_part000_limit153121")
INDEX_FILENAME = "bm25_index.pkl"
CHUNK_STORE_FILENAME = "chunk_store.parquet"
STATS_FILENAME = "bm25_stats.json"

_TERM_RE = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*", flags=re.UNICODE)
_CJK_RE = re.compile(r"^[\u3400-\u9fff]+$")


def medical_tokenize(text: str | None) -> list[str]:
    """Tokenize biomedical English conservatively and retain useful CJK n-grams."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    tokens: list[str] = []
    for match in _TERM_RE.finditer(normalized):
        token = match.group(0).strip("-./")
        if not token:
            continue
        tokens.append(token)
        if _CJK_RE.fullmatch(token):
            for width in (2, 3):
                if len(token) >= width:
                    tokens.extend(token[start : start + width] for start in range(len(token) - width + 1))
        elif any(separator in token for separator in "-./"):
            tokens.extend(part for part in re.split(r"[-./]+", token) if part)
    return tokens


def keyword_query_text(keyword_query: dict[str, Any] | str | None, fallback: str = "") -> str:
    if isinstance(keyword_query, str):
        return keyword_query.strip() or fallback
    if not isinstance(keyword_query, dict):
        return fallback
    values: list[str] = []
    for key in ("required_terms", "optional_terms"):
        raw = keyword_query.get(key, [])
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, Iterable):
            values.extend(str(item) for item in raw if item)
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.casefold().strip()
        if key and key not in seen:
            deduped.append(value.strip())
            seen.add(key)
    return " ".join(deduped) or fallback


def _selected_parts(
    manifest_path: Path,
    chunks_dir: Path,
    max_parts: int | None,
) -> list[dict[str, Any]]:
    if max_parts is not None and max_parts <= 0:
        raise ValueError("max_parts must be positive when provided")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            part_file = Path(row["part_file"])
            if not part_file.is_absolute():
                manifest_relative = manifest_path.parent / part_file.name
                chunks_relative = chunks_dir / part_file.name
                part_file = chunks_relative if chunks_relative.exists() else manifest_relative
            if not part_file.exists():
                raise FileNotFoundError(f"Missing chunk parquet: {part_file}")
            rows.append(
                {
                    "part_id": int(row["part_id"]),
                    "part_file": part_file,
                    "chunk_count": int(row.get("chunk_count") or 0),
                }
            )
            if max_parts is not None and len(rows) >= max_parts:
                break
    if not rows:
        raise ValueError(f"No chunk parts selected from manifest: {manifest_path}")
    return rows


def _store_columns(schema_names: list[str]) -> list[str]:
    preferred = ["chunk_id", "doc_id", "text", *DEFAULT_METADATA_FIELDS]
    return list(dict.fromkeys(name for name in preferred if name in schema_names))


def build_bm25_index(
    manifest_path: str | Path,
    chunks_dir: str | Path,
    output_dir: str | Path,
    *,
    output_prefix: str,
    max_parts: int | None = None,
    max_chunks: int | None = None,
    force: bool = False,
    k1: float = 1.5,
    b: float = 0.75,
    parquet_batch_size: int = 2048,
) -> dict[str, Any]:
    """Build a persistent BM25 inverted index and aligned Parquet chunk store."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.time()
    manifest_path = Path(manifest_path).resolve()
    chunks_dir = Path(chunks_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing chunks manifest: {manifest_path}")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive when provided")
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 parameters require k1 > 0 and 0 <= b <= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / INDEX_FILENAME
    chunk_store_path = output_dir / CHUNK_STORE_FILENAME
    stats_path = output_dir / STATS_FILENAME
    temporary_store = output_dir / f".{CHUNK_STORE_FILENAME}.tmp"
    existing = [path for path in (index_path, chunk_store_path, stats_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"BM25 output already exists ({names}); pass --force to replace this index")
    if force:
        for path in (index_path, chunk_store_path, stats_path, temporary_store):
            if path.exists():
                path.unlink()

    parts = _selected_parts(manifest_path, chunks_dir, max_parts)
    postings_docs: dict[str, array] = defaultdict(lambda: array("I"))
    postings_tfs: dict[str, array] = defaultdict(lambda: array("I"))
    doc_lengths = array("I")
    selected_part_ids: list[int] = []
    total_tokens = 0
    row_index = 0
    writer: pq.ParquetWriter | None = None

    try:
        for part in parts:
            if max_chunks is not None and row_index >= max_chunks:
                break
            parquet_file = pq.ParquetFile(part["part_file"])
            schema_names = parquet_file.schema.names
            required = {"chunk_id", "doc_id", "text"}
            missing = required - set(schema_names)
            if missing:
                raise ValueError(f"{part['part_file']} missing required columns: {sorted(missing)}")
            columns = _store_columns(schema_names)
            selected_part_ids.append(part["part_id"])
            for batch in parquet_file.iter_batches(batch_size=parquet_batch_size, columns=columns):
                if max_chunks is not None:
                    remaining = max_chunks - row_index
                    if remaining <= 0:
                        break
                    if batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                rows = batch.to_pylist()
                batch_start_index = row_index
                for row in rows:
                    text = str(row.get("text") or "")
                    term_counts = Counter(medical_tokenize(text))
                    doc_length = sum(term_counts.values())
                    doc_lengths.append(doc_length)
                    total_tokens += doc_length
                    for term, term_frequency in term_counts.items():
                        postings_docs[term].append(row_index)
                        postings_tfs[term].append(term_frequency)
                    row_index += 1
                if rows:
                    table = pa.Table.from_batches([batch]).append_column(
                        "_bm25_doc_index",
                        pa.array(range(batch_start_index, row_index), type=pa.int64()),
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(temporary_store, table.schema, compression="zstd")
                    writer.write_table(table)
                if max_chunks is not None and row_index >= max_chunks:
                    break
    finally:
        if writer is not None:
            writer.close()

    if row_index == 0 or writer is None:
        if temporary_store.exists():
            temporary_store.unlink()
        raise ValueError("No chunks were indexed")

    index_data = {
        "version": 1,
        "algorithm": "bm25_okapi",
        "k1": float(k1),
        "b": float(b),
        "document_count": row_index,
        "average_document_length": total_tokens / row_index,
        "document_lengths": doc_lengths,
        "postings": {
            term: (postings_docs[term], postings_tfs[term])
            for term in postings_docs
        },
        "tokenizer": "medical_tokenize_v1",
        "output_prefix": output_prefix,
    }
    with index_path.open("wb") as handle:
        pickle.dump(index_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_store.replace(chunk_store_path)

    elapsed = time.time() - started
    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_prefix": output_prefix,
        "algorithm": "bm25_okapi",
        "tokenizer": "medical_tokenize_v1",
        "k1": float(k1),
        "b": float(b),
        "source_manifest": str(manifest_path),
        "selected_part_ids": selected_part_ids,
        "selected_part_count": len(selected_part_ids),
        "max_parts": max_parts,
        "max_chunks": max_chunks,
        "document_count": row_index,
        "total_tokens": total_tokens,
        "average_document_length": total_tokens / row_index,
        "vocabulary_size": len(postings_docs),
        "index_path": str(index_path),
        "chunk_store_path": str(chunk_store_path),
        "index_size_bytes": index_path.stat().st_size,
        "chunk_store_size_bytes": chunk_store_path.stat().st_size,
        "elapsed_seconds": elapsed,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


class BM25Retriever:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir).resolve()
        self.index_path = self.index_dir / INDEX_FILENAME
        self.chunk_store_path = self.index_dir / CHUNK_STORE_FILENAME
        if not self.index_path.exists() or not self.chunk_store_path.exists():
            raise FileNotFoundError(
                f"BM25 index is incomplete at {self.index_dir}. "
                "Run scripts/18_build_bm25_index.py first."
            )
        with self.index_path.open("rb") as handle:
            self.data: dict[str, Any] = pickle.load(handle)
        self.document_count = int(self.data["document_count"])
        self.average_document_length = float(self.data["average_document_length"])
        self.k1 = float(self.data["k1"])
        self.b = float(self.data["b"])

    def _scores(self, query_tokens: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        query_counts = Counter(query_tokens)
        doc_lengths = self.data["document_lengths"]
        postings = self.data["postings"]
        for term, query_frequency in query_counts.items():
            posting = postings.get(term)
            if posting is None:
                continue
            doc_ids, term_frequencies = posting
            document_frequency = len(doc_ids)
            idf = math.log(1.0 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for doc_index, term_frequency in zip(doc_ids, term_frequencies):
                length_norm = 1.0 - self.b + self.b * doc_lengths[doc_index] / max(self.average_document_length, 1e-9)
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * length_norm
                scores[int(doc_index)] += query_frequency * idf * numerator / denominator
        return scores

    def _read_rows(self, doc_indices: list[int]) -> dict[int, dict[str, Any]]:
        import pyarrow.dataset as ds

        dataset = ds.dataset(self.chunk_store_path, format="parquet")
        table = dataset.to_table(filter=ds.field("_bm25_doc_index").isin(doc_indices))
        return {int(row["_bm25_doc_index"]): row for row in table.to_pylist()}

    def search(
        self,
        query: str | dict[str, Any],
        *,
        top_k: int = 10,
        fallback_query: str = "",
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query_text = keyword_query_text(query, fallback=fallback_query)
        tokens = medical_tokenize(query_text)
        if not tokens:
            return []
        scores = self._scores(tokens)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        rows = self._read_rows([doc_index for doc_index, _ in ranked]) if ranked else {}
        results: list[dict[str, Any]] = []
        for rank, (doc_index, score) in enumerate(ranked, start=1):
            row = rows.get(doc_index, {})
            metadata = {
                key: clean_metadata_value(value)
                for key, value in row.items()
                if key not in {"_bm25_doc_index", "chunk_id", "doc_id", "text"}
            }
            results.append(
                {
                    "chunk_id": str(row.get("chunk_id") or ""),
                    "doc_id": str(row.get("doc_id") or metadata.get("doc_id") or ""),
                    "text": str(row.get("text") or ""),
                    "metadata": metadata,
                    "bm25_score": float(score),
                    "keyword_rank": rank,
                    "query_text": query_text,
                }
            )
        return results
