from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from medical_rag.common.pmc import extract_article_id, extract_pub_date, first_text, setup_tee, text_or_empty

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_PREFIX = "limit153121"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_WHOLE_DOC_TOKEN_LIMIT = 512
SAMPLE_LIMIT = 1000

CHUNK_COLUMNS = [
    "chunk_id",
    "text",
    "doc_id",
    "chunk_index",
    "total_chunks",
    "source_title",
    "token_count",
    "source_file",
    "journal",
    "pub_date",
    "pub_year",
    "pmid",
    "pmcid",
    "article_type",
    "section_title",
    "section_title_norm",
    "split_strategy",
    "quality_decision",
    "title_missing",
    "body_missing",
    "chunk_char_len",
    "section_index",
    "section_chunk_index",
]

EXCLUDED_COLUMNS = [
    "doc_id",
    "pmid",
    "pmcid",
    "title",
    "source_file",
    "excluded_reason",
    "has_title",
    "has_abstract",
    "has_body",
    "title_token_len",
    "abstract_token_len",
    "body_token_len",
    "fallback_available",
    "note",
]


def ensure_project_hf_cache(project_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str(project_dir / "artifacts/models/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_local_tokenizer(project_dir: Path) -> Path | str:
    snapshot_root = (
        project_dir
        / "artifacts/models/huggingface"
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )
    if snapshot_root.exists():
        for snapshot in sorted(snapshot_root.iterdir()):
            if (snapshot / "tokenizer.json").exists() and (snapshot / "config.json").exists():
                return snapshot
    return MODEL


def ensure_output_dirs() -> None:
    for rel in [
        "artifacts/datasets/chunks",
        "artifacts/metrics/t007_chunking",
        "reports/samples",
        "reports/formal",
        "logs",
    ]:
        Path(rel).mkdir(parents=True, exist_ok=True)


def progress_line(
    label: str,
    current: int,
    total: int,
    started_at: float,
    width: int = 24,
) -> str:
    ratio = 0.0 if total <= 0 else min(1.0, current / total)
    filled = int(width * ratio)
    elapsed = max(0.001, time.perf_counter() - started_at)
    rate = current / elapsed if current > 0 else 0.0
    remaining = max(0, total - current)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"{label} [{bar}] {current}/{total} "
        f"({ratio:.1%}) rate={rate:.1f}/s eta={eta_seconds/60:.1f}min"
    )


def token_len(tokenizer, text: str) -> int:
    if not text or not str(text).strip():
        return 0
    return len(tokenizer.encode(str(text), add_special_tokens=True, truncation=False))


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def normalize_section_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def short_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


def normalize_pmcid(pmcid: str) -> str:
    value = str(pmcid or "").strip()
    if not value:
        return ""
    return value if value.upper().startswith("PMC") else f"PMC{value}"


def choose_doc_id(pmcid: str, pmid: str, source_file: str) -> str:
    pmcid_norm = normalize_pmcid(pmcid)
    if pmcid_norm:
        return f"PMCID:{pmcid_norm}"
    if str(pmid or "").strip():
        return f"PMID:{str(pmid).strip()}"
    return f"SOURCE:{short_hash(source_file)}"


def fallback_available(title: str, abstract: str) -> str:
    if abstract and abstract.strip():
        return "abstract_available"
    if title and title.strip():
        return "title_available"
    return "none"


def encoding_issue_note(*texts: str) -> str:
    joined = " ".join(texts)
    if not joined:
        return ""
    replacement_count = joined.count("\ufffd")
    control_count = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", joined))
    bad_count = replacement_count + control_count
    if bad_count == 0:
        return ""
    density = bad_count / max(1, len(joined))
    if density > 0.01 or bad_count >= 20:
        return f"severe_encoding_issue bad_chars={bad_count} density={density:.4f}"
    return f"encoding_warning bad_chars={bad_count} density={density:.4f}"


def read_soup(path: Path) -> BeautifulSoup:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return BeautifulSoup(raw, "lxml-xml")


def extract_base_document(path: Path, source_root: Path, tokenizer) -> dict[str, object]:
    soup = read_soup(path)
    article = soup.find("article")
    body_node = soup.find("body")
    article_type = article.get("article-type", "") if article else ""
    title = first_text(soup, "article-title")
    abstract = first_text(soup, "abstract")
    body = text_or_empty(body_node)
    journal = first_text(soup, "journal-title") or first_text(soup, "journal-id")
    pub_date, pub_year = extract_pub_date(soup)
    pmid = extract_article_id(soup, "pmid")
    pmcid = normalize_pmcid(extract_article_id(soup, "pmc"))
    source_file = str(path.relative_to(source_root))
    doc_id = choose_doc_id(pmcid, pmid, source_file)
    return {
        "path": path,
        "soup": soup,
        "body_node": body_node,
        "doc_id": doc_id,
        "source_file": source_file,
        "title": title,
        "abstract": abstract,
        "body": body,
        "journal": journal,
        "pub_date": pub_date,
        "pub_year": pub_year,
        "pmid": pmid,
        "pmcid": pmcid,
        "article_type": article_type,
        "title_token_len": token_len(tokenizer, title),
        "abstract_token_len": token_len(tokenizer, abstract),
        "body_token_len": token_len(tokenizer, body),
    }


def scan_duplicate_ids(xml_files: list[Path]) -> tuple[Counter[str], Counter[str]]:
    pmcid_counter: Counter[str] = Counter()
    pmid_counter: Counter[str] = Counter()
    for path in xml_files:
        soup = read_soup(path)
        pmcid = normalize_pmcid(extract_article_id(soup, "pmc"))
        pmid = extract_article_id(soup, "pmid")
        if pmcid:
            pmcid_counter[pmcid] += 1
        if pmid:
            pmid_counter[pmid] += 1
    return pmcid_counter, pmid_counter


def direct_title(sec) -> str:
    title_node = sec.find("title", recursive=False)
    return text_or_empty(title_node)


def section_text_without_direct_title(sec) -> str:
    title_node = sec.find("title", recursive=False)
    if title_node is None:
        return text_or_empty(sec)
    title_node.extract()
    return text_or_empty(sec)


def extract_sections(body_node) -> list[dict[str, object]]:
    if body_node is None:
        return []
    sections: list[dict[str, object]] = []
    top_level = body_node.find_all("sec", recursive=False)
    candidates = top_level if top_level else body_node.find_all("sec")
    for idx, sec in enumerate(candidates):
        title = direct_title(sec)
        if not title:
            continue
        # Work on a copy so later section traversal is not affected.
        sec_copy = BeautifulSoup(str(sec), "lxml-xml").find("sec")
        section_text = section_text_without_direct_title(sec_copy) if sec_copy else ""
        if section_text.strip():
            sections.append(
                {
                    "section_index": idx,
                    "section_title": title,
                    "section_title_norm": normalize_section_title(title),
                    "section_text": section_text,
                }
            )
    return sections


def split_body_text(text: str, splitter: RecursiveCharacterTextSplitter, tokenizer, chunk_size: int) -> list[str]:
    if not text or not text.strip():
        return []
    if token_len(tokenizer, text) <= chunk_size:
        return [normalize_space(text)]
    return [normalize_space(chunk) for chunk in splitter.split_text(text) if chunk and chunk.strip()]


def build_chunk_text(title_or_fallback: str, section_title: str, body_text: str) -> str:
    return f"Title: {title_or_fallback}\nSection: {section_title}\nText:\n{body_text.strip()}".strip()


def split_body_for_final_token_limit(
    body_text: str,
    title_or_fallback: str,
    section_title: str,
    tokenizer,
    chunk_size: int,
    chunk_overlap: int,
    final_token_limit: int,
) -> list[str]:
    text = normalize_space(body_text)
    if not text:
        return []

    def final_len(piece: str) -> int:
        return token_len(tokenizer, build_chunk_text(title_or_fallback, section_title, piece))

    if final_len(text) <= final_token_limit:
        return [text]

    prefix_tokens = token_len(tokenizer, build_chunk_text(title_or_fallback, section_title, ""))
    budget = min(chunk_size, max(32, final_token_limit - prefix_tokens - 8))
    overlap = min(chunk_overlap, max(0, budget // 5))

    def make_splitter(size: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=min(overlap_tokens, max(0, size - 1)),
            length_function=lambda value: token_len(tokenizer, value),
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

    pieces = [normalize_space(piece) for piece in make_splitter(budget, overlap).split_text(text) if piece.strip()]
    if not pieces:
        pieces = [text]

    for size in [budget, max(24, budget // 2), max(16, budget // 3), max(12, budget // 4)]:
        next_pieces: list[str] = []
        changed = False
        splitter = make_splitter(size, min(overlap, max(0, size // 5)))
        for piece in pieces:
            if final_len(piece) <= final_token_limit:
                next_pieces.append(piece)
                continue
            subpieces = [normalize_space(sub) for sub in splitter.split_text(piece) if sub.strip()]
            if not subpieces:
                next_pieces.append(piece)
                continue
            if len(subpieces) > 1 or subpieces[0] != piece:
                changed = True
            next_pieces.extend(subpieces)
        pieces = next_pieces
        if all(final_len(piece) <= final_token_limit for piece in pieces):
            break
        if not changed and size <= 16:
            break
    return pieces


def quality_decision_for(doc: dict[str, object], pmid_counter: Counter[str], encoding_note: str) -> str:
    title_missing = not str(doc["title"]).strip()
    abstract_missing = not str(doc["abstract"]).strip()
    body = str(doc["body"])
    body_tokens = int(doc["body_token_len"])
    pmid = str(doc["pmid"])
    if encoding_note.startswith("severe_encoding_issue"):
        return "need_review"
    if pmid and pmid_counter.get(pmid, 0) > 1:
        return "need_review"
    if title_missing or abstract_missing or body_tokens < 50:
        return "keep_with_warning"
    if len(body.strip()) < 200:
        return "keep_with_warning"
    return "keep"


def excluded_row(doc: dict[str, object], reason: str, note: str = "") -> dict[str, object]:
    return {
        "doc_id": doc.get("doc_id", ""),
        "pmid": doc.get("pmid", ""),
        "pmcid": doc.get("pmcid", ""),
        "title": doc.get("title", ""),
        "source_file": doc.get("source_file", ""),
        "excluded_reason": reason,
        "has_title": bool(str(doc.get("title", "")).strip()),
        "has_abstract": bool(str(doc.get("abstract", "")).strip()),
        "has_body": bool(str(doc.get("body", "")).strip()),
        "title_token_len": doc.get("title_token_len", 0),
        "abstract_token_len": doc.get("abstract_token_len", 0),
        "body_token_len": doc.get("body_token_len", 0),
        "fallback_available": fallback_available(str(doc.get("title", "")), str(doc.get("abstract", ""))),
        "note": note,
    }


def build_chunks_for_doc(
    doc: dict[str, object],
    splitter: RecursiveCharacterTextSplitter,
    tokenizer,
    chunk_size: int,
    chunk_overlap: int,
    whole_doc_token_limit: int,
    pmid_counter: Counter[str],
) -> tuple[list[dict[str, object]], str]:
    title = str(doc["title"])
    body = str(doc["body"])
    doc_id = str(doc["doc_id"])
    title_missing = not title.strip()
    title_or_fallback = title.strip() if title.strip() else f"[Missing title: {doc_id}]"
    title_body = "\n\n".join(part for part in [title_or_fallback, body] if part and part.strip())
    sections = extract_sections(doc["body_node"])
    title_body_tokens = token_len(tokenizer, title_body)
    encoding_note = encoding_issue_note(title, str(doc["abstract"]), body)
    quality_decision = quality_decision_for(doc, pmid_counter, encoding_note)
    if title_body_tokens <= whole_doc_token_limit:
        split_strategy = "whole_document_under_512"
        section_entries = [
            {
                "section_index": 0,
                "section_title": "whole_document",
                "section_title_norm": "whole_document",
                "chunks": split_body_for_final_token_limit(
                    body,
                    title_or_fallback,
                    "whole_document",
                    tokenizer,
                    chunk_size,
                    chunk_overlap,
                    whole_doc_token_limit,
                ),
            }
        ]
    elif sections:
        split_strategy = "semantic_section"
        section_entries = []
        for section in sections:
            section_chunks = split_body_for_final_token_limit(
                str(section["section_text"]),
                title_or_fallback,
                str(section["section_title"]),
                tokenizer,
                chunk_size,
                chunk_overlap,
                whole_doc_token_limit,
            )
            if not section_chunks:
                continue
            section_entries.append(
                {
                    "section_index": int(section["section_index"]),
                    "section_title": str(section["section_title"]),
                    "section_title_norm": str(section["section_title_norm"]),
                    "chunks": section_chunks,
                }
            )
        if not section_entries:
            split_strategy = "recursive_fallback_no_section"
            section_entries = [
                {
                    "section_index": 0,
                    "section_title": "recursive_fallback",
                    "section_title_norm": "recursive_fallback",
                    "chunks": split_body_for_final_token_limit(
                        title_body,
                        title_or_fallback,
                        "recursive_fallback",
                        tokenizer,
                        chunk_size,
                        chunk_overlap,
                        whole_doc_token_limit,
                    ),
                }
            ]
    else:
        split_strategy = "recursive_fallback_no_section"
        section_entries = [
            {
                "section_index": 0,
                "section_title": "recursive_fallback",
                "section_title_norm": "recursive_fallback",
                "chunks": split_body_for_final_token_limit(
                    title_body,
                    title_or_fallback,
                    "recursive_fallback",
                    tokenizer,
                    chunk_size,
                    chunk_overlap,
                    whole_doc_token_limit,
                ),
            }
        ]

    rows: list[dict[str, object]] = []
    for section in section_entries:
        for section_chunk_index, chunk_body in enumerate(section["chunks"]):
            if not str(chunk_body).strip():
                continue
            text = build_chunk_text(title_or_fallback, str(section["section_title"]), str(chunk_body))
            rows.append(
                {
                    "chunk_id": "",
                    "text": text,
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "total_chunks": 0,
                    "source_title": title,
                    "token_count": token_len(tokenizer, text),
                    "source_file": doc["source_file"],
                    "journal": doc["journal"],
                    "pub_date": doc["pub_date"],
                    "pub_year": doc["pub_year"],
                    "pmid": doc["pmid"],
                    "pmcid": doc["pmcid"],
                    "article_type": doc["article_type"],
                    "section_title": section["section_title"],
                    "section_title_norm": section["section_title_norm"],
                    "split_strategy": split_strategy,
                    "quality_decision": quality_decision,
                    "title_missing": title_missing,
                    "body_missing": False,
                    "chunk_char_len": len(text),
                    "section_index": int(section["section_index"]),
                    "section_chunk_index": section_chunk_index,
                }
            )
    total = len(rows)
    for idx, row in enumerate(rows):
        row["chunk_index"] = idx
        row["total_chunks"] = total
        row["chunk_id"] = f"{doc_id}::chunk_{idx:05d}"
    return rows, encoding_note


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    if not rows:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in names} for row in rows)


def part_file(prefix: str, part_id: int) -> Path:
    return Path(f"artifacts/datasets/chunks/pmc_chunks_{prefix}_part{part_id:03d}.parquet")


def prefix_paths(prefix: str) -> dict[str, Path]:
    return {
        "manifest": Path(f"artifacts/datasets/chunks/pmc_chunks_{prefix}_manifest.csv"),
        "excluded": Path(f"artifacts/metrics/t007_chunking/excluded_documents_{prefix}.csv"),
        "summary": Path(f"artifacts/metrics/t007_chunking/chunk_summary_{prefix}.csv"),
        "route_summary": Path(f"artifacts/metrics/t007_chunking/chunk_route_summary_{prefix}.csv"),
        "token_stats": Path(f"artifacts/metrics/t007_chunking/chunk_token_length_stats_{prefix}.csv"),
        "section_top": Path(f"artifacts/metrics/t007_chunking/chunk_section_title_top80_{prefix}.csv"),
        "quality_flags": Path(f"artifacts/metrics/t007_chunking/chunk_quality_flags_{prefix}.csv"),
        "processing_stats": Path(f"artifacts/metrics/t007_chunking/chunk_processing_stats_{prefix}.json"),
        "preview_md": Path(f"reports/samples/chunk_preview_{prefix}.md"),
        "preview_jsonl": Path(f"reports/samples/chunk_preview_{prefix}_sample.jsonl"),
        "report": Path(f"reports/formal/文档解析与分割质量验证_{prefix}.md"),
    }


def clear_prefix_outputs(prefix: str) -> None:
    for path in Path("artifacts/datasets/chunks").glob(f"pmc_chunks_{prefix}_part*.parquet"):
        path.unlink()
    for path in prefix_paths(prefix).values():
        if path.exists():
            path.unlink()


def stats_for(values: list[int], label: str) -> dict[str, object]:
    if not values:
        return {
            "metric": label,
            "count": 0,
            "mean": 0,
            "median": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    s = pd.Series(values, dtype="int64")
    return {
        "metric": label,
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": int(s.max()),
    }


def markdown_table_from_rows(rows: list[dict[str, object]], max_rows: int | None = None) -> str:
    if not rows:
        return ""
    shown = rows[:max_rows] if max_rows is not None else rows
    columns = list(shown[0].keys())

    def cell(value: object) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |" for row in shown]
    return "\n".join([header, sep, *body])


def validation_summary(part_paths: list[Path], excluded_path: Path) -> dict[str, object]:
    token_counts: list[int] = []
    chunks_per_doc: dict[str, int] = defaultdict(int)
    max_index: dict[str, int] = defaultdict(lambda: -1)
    total_chunks_values: dict[str, set[int]] = defaultdict(set)
    section_counter: Counter[str] = Counter()
    strategy_doc_ids: dict[str, set[str]] = defaultdict(set)
    strategy_chunk_counter: Counter[str] = Counter()
    quality_counter: Counter[str] = Counter()
    chunk_ids: set[str] = set()
    duplicate_chunk_ids = 0
    doc_id_missing = 0
    source_title_missing = 0
    pmcid_missing = 0
    source_file_missing = 0
    empty_text_chunks = 0
    short_chunks = 0
    garbled_chunks = 0
    token_over_512 = 0
    token_le_zero = 0
    semantic_without_section_title = 0
    recursive_multi_doc_count = 0

    for path in part_paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for row in df.itertuples(index=False):
            chunk_id = str(row.chunk_id)
            if chunk_id in chunk_ids:
                duplicate_chunk_ids += 1
            chunk_ids.add(chunk_id)
            doc_id = str(row.doc_id)
            if not doc_id.strip():
                doc_id_missing += 1
            chunks_per_doc[doc_id] += 1
            max_index[doc_id] = max(max_index[doc_id], int(row.chunk_index))
            total_chunks_values[doc_id].add(int(row.total_chunks))
            token = int(row.token_count)
            token_counts.append(token)
            if token > 512:
                token_over_512 += 1
            if token <= 0:
                token_le_zero += 1
            text = str(row.text)
            if not text.strip():
                empty_text_chunks += 1
            if token < 20:
                short_chunks += 1
            if "\ufffd" in text:
                garbled_chunks += 1
            if not str(row.source_title).strip():
                source_title_missing += 1
            if not str(row.pmcid).strip():
                pmcid_missing += 1
            if not str(row.source_file).strip():
                source_file_missing += 1
            strategy = str(row.split_strategy)
            strategy_doc_ids[strategy].add(doc_id)
            strategy_chunk_counter[strategy] += 1
            quality_counter[str(row.quality_decision)] += 1
            section_title = str(row.section_title)
            section_counter[section_title] += 1
            if strategy == "semantic_section" and not section_title.strip():
                semantic_without_section_title += 1

    non_continuous_docs = 0
    total_chunks_mismatch_docs = 0
    max_index_mismatch_docs = 0
    for doc_id, count in chunks_per_doc.items():
        if max_index[doc_id] != count - 1:
            non_continuous_docs += 1
            max_index_mismatch_docs += 1
        if total_chunks_values[doc_id] != {count}:
            total_chunks_mismatch_docs += 1

    excluded_df = pd.read_csv(excluded_path, dtype=str, keep_default_na=False) if excluded_path.exists() else pd.DataFrame()
    excluded_reason_counts = excluded_df["excluded_reason"].value_counts().to_dict() if not excluded_df.empty else {}
    recursive_docs = strategy_doc_ids.get("recursive_fallback_no_section", set())
    if recursive_docs:
        recursive_multi_doc_count = sum(1 for doc_id in recursive_docs if chunks_per_doc[doc_id] > 1)

    return {
        "token_counts": token_counts,
        "chunks_per_doc": list(chunks_per_doc.values()),
        "section_counter": section_counter,
        "strategy_doc_counts": {k: len(v) for k, v in strategy_doc_ids.items()},
        "strategy_chunk_counts": dict(strategy_chunk_counter),
        "quality_counts": dict(quality_counter),
        "excluded_reason_counts": excluded_reason_counts,
        "excluded_count": int(excluded_df.shape[0]),
        "chunked_documents": len(chunks_per_doc),
        "total_chunks": len(token_counts),
        "token_over_512": token_over_512,
        "token_le_zero": token_le_zero,
        "empty_text_chunks": empty_text_chunks,
        "short_chunks": short_chunks,
        "garbled_chunks": garbled_chunks,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "doc_id_missing": doc_id_missing,
        "source_title_missing": source_title_missing,
        "pmcid_missing": pmcid_missing,
        "source_file_missing": source_file_missing,
        "non_continuous_docs": non_continuous_docs,
        "total_chunks_mismatch_docs": total_chunks_mismatch_docs,
        "max_index_mismatch_docs": max_index_mismatch_docs,
        "semantic_without_section_title": semantic_without_section_title,
        "recursive_multi_doc_count": recursive_multi_doc_count,
    }


def write_sample_jsonl(part_paths: list[Path], path: Path, limit: int = SAMPLE_LIMIT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for part in part_paths:
            if written >= limit:
                break
            df = pd.read_parquet(part)
            for rec in df.head(max(0, limit - written)).to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if written >= limit:
                    break


def collect_preview_samples(part_paths: list[Path], excluded_path: Path) -> dict[str, object]:
    samples: dict[str, object] = {}
    multi_doc_id = ""
    multi_rows: list[dict[str, object]] = []
    for part in part_paths:
        df = pd.read_parquet(part)
        for strategy in ["whole_document_under_512", "semantic_section", "recursive_fallback_no_section"]:
            if strategy not in samples:
                hit = df[df["split_strategy"] == strategy].head(1)
                if not hit.empty:
                    samples[strategy] = hit.iloc[0].to_dict()
        if "title_missing" not in samples:
            hit = df[df["title_missing"].astype(bool)].head(1)
            if not hit.empty:
                samples["title_missing"] = hit.iloc[0].to_dict()
        if not multi_doc_id:
            counts = df.groupby("doc_id").size()
            candidates = counts[counts > 1]
            if not candidates.empty:
                multi_doc_id = str(candidates.index[0])
                multi_rows = df[df["doc_id"] == multi_doc_id].head(4).to_dict(orient="records")
        if len(samples) >= 4 and multi_doc_id:
            break
    if multi_rows:
        samples["multi_chunk_rows"] = multi_rows
    if excluded_path.exists():
        excluded_df = pd.read_csv(excluded_path, dtype=str, keep_default_na=False)
        hit = excluded_df[excluded_df["excluded_reason"] == "no_body_for_fulltext"].head(1)
        if not hit.empty:
            samples["excluded_no_body"] = hit.iloc[0].to_dict()
    return samples


def format_chunk_sample(row: dict[str, object]) -> str:
    text = normalize_space(str(row.get("text", "")))[:800]
    return f"""
- chunk_id: `{row.get('chunk_id', '')}`
- doc_id: `{row.get('doc_id', '')}`
- split_strategy: `{row.get('split_strategy', '')}`
- chunk_index / total_chunks: `{row.get('chunk_index', '')}` / `{row.get('total_chunks', '')}`
- source_title: {row.get('source_title', '')}
- journal: {row.get('journal', '')}
- pub_year: {row.get('pub_year', '')}
- pmid: `{row.get('pmid', '')}`
- pmcid: `{row.get('pmcid', '')}`
- section_title: {row.get('section_title', '')}
- token_count: `{row.get('token_count', '')}`
- text 前 800 字符:

```text
{text}
```
""".strip()


def write_preview_markdown(path: Path, samples: dict[str, object], prefix: str) -> None:
    lines = [
        f"# Chunk 预览（{prefix}）",
        "",
        "## whole_document_under_512 样本",
        format_chunk_sample(samples["whole_document_under_512"])
        if "whole_document_under_512" in samples
        else "未在本次样本中命中。",
        "",
        "## semantic_section 样本",
        format_chunk_sample(samples["semantic_section"]) if "semantic_section" in samples else "未在本次样本中命中。",
        "",
        "## recursive_fallback_no_section 样本",
        format_chunk_sample(samples["recursive_fallback_no_section"])
        if "recursive_fallback_no_section" in samples
        else "未在本次样本中命中。",
        "",
        "## 多 chunk 文献样本",
    ]
    for row in samples.get("multi_chunk_rows", []):
        lines.extend(["", format_chunk_sample(row)])
    if not samples.get("multi_chunk_rows"):
        lines.append("未在本次样本中命中。")
    lines.extend(
        [
            "",
            "## title 缺失但保留的样本",
            format_chunk_sample(samples["title_missing"]) if "title_missing" in samples else "未在本次样本中命中 title 缺失但保留的 chunk。",
            "",
            "## excluded no_body 样本",
        ]
    )
    excluded = samples.get("excluded_no_body")
    if isinstance(excluded, dict):
        lines.append(
            "\n".join(
                [
                    f"- doc_id: `{excluded.get('doc_id', '')}`",
                    f"- pmid: `{excluded.get('pmid', '')}`",
                    f"- pmcid: `{excluded.get('pmcid', '')}`",
                    f"- source_file: `{excluded.get('source_file', '')}`",
                    f"- excluded_reason: `{excluded.get('excluded_reason', '')}`",
                    f"- fallback_available: `{excluded.get('fallback_available', '')}`",
                    f"- title: {excluded.get('title', '')}",
                ]
            )
        )
    else:
        lines.append("未在本次样本中命中 no_body_for_fulltext excluded 文献。")
    if "semantic_section" in samples:
        row = samples["semantic_section"]
        lines.extend(
            [
                "",
                "## chunk metadata 示例",
                "```json",
                json.dumps({k: v for k, v in row.items() if k != "text"}, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    lines.extend(["", "## 一个 doc_id 下连续多个 chunk 的示例"])
    if samples.get("multi_chunk_rows"):
        for row in samples["multi_chunk_rows"]:
            lines.append(f"- `{row.get('chunk_id')}` section=`{row.get('section_title')}` token_count=`{row.get('token_count')}`")
    else:
        lines.append("未在本次样本中命中。")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_tables_and_reports(
    args,
    paths: dict[str, Path],
    manifest_rows: list[dict[str, object]],
    part_paths: list[Path],
    stats: dict[str, object],
    elapsed_seconds: float,
    tokenizer_ref: Path | str,
) -> None:
    total_docs = int(args.selected_xml_count)
    token_counts = stats["token_counts"]
    chunks_per_doc = stats["chunks_per_doc"]
    total_chunks = int(stats["total_chunks"])
    excluded_count = int(stats["excluded_count"])
    processed_documents = total_docs
    chunked_documents = int(stats["chunked_documents"])
    over512_rate = 0.0 if total_chunks == 0 else int(stats["token_over_512"]) / total_chunks
    summary_rows = [
        {"metric": "original_xml", "value": total_docs},
        {"metric": "processed_documents", "value": processed_documents},
        {"metric": "chunked_documents", "value": chunked_documents},
        {"metric": "excluded_documents", "value": excluded_count},
        {"metric": "total_chunks", "value": total_chunks},
        {"metric": "token_count_over_512", "value": int(stats["token_over_512"])},
        {"metric": "token_count_over_512_rate", "value": over512_rate},
        {"metric": "token_count_le_zero", "value": int(stats["token_le_zero"])},
        {"metric": "empty_text_chunks", "value": int(stats["empty_text_chunks"])},
        {"metric": "duplicate_chunk_ids", "value": int(stats["duplicate_chunk_ids"])},
        {"metric": "doc_id_missing", "value": int(stats["doc_id_missing"])},
        {"metric": "source_title_missing", "value": int(stats["source_title_missing"])},
        {"metric": "pmcid_missing", "value": int(stats["pmcid_missing"])},
        {"metric": "source_file_missing", "value": int(stats["source_file_missing"])},
        {"metric": "non_continuous_doc_count", "value": int(stats["non_continuous_docs"])},
        {"metric": "total_chunks_mismatch_doc_count", "value": int(stats["total_chunks_mismatch_docs"])},
        {"metric": "max_index_mismatch_doc_count", "value": int(stats["max_index_mismatch_docs"])},
        {"metric": "elapsed_seconds", "value": f"{elapsed_seconds:.3f}"},
    ]
    for reason, count in sorted(stats["excluded_reason_counts"].items()):
        summary_rows.append({"metric": f"excluded_reason_{reason}", "value": int(count)})
    write_csv(paths["summary"], summary_rows)

    route_rows = []
    for strategy in sorted(set(stats["strategy_doc_counts"]) | set(stats["strategy_chunk_counts"])):
        route_rows.append(
            {
                "split_strategy": strategy,
                "document_count": int(stats["strategy_doc_counts"].get(strategy, 0)),
                "chunk_count": int(stats["strategy_chunk_counts"].get(strategy, 0)),
            }
        )
    for reason, count in sorted(stats["excluded_reason_counts"].items()):
        route_rows.append({"split_strategy": reason, "document_count": int(count), "chunk_count": 0})
    write_csv(paths["route_summary"], route_rows)

    token_stat_rows = [
        stats_for(chunks_per_doc, "chunks_per_doc"),
        stats_for(token_counts, "chunk_token_count"),
    ]
    write_csv(paths["token_stats"], token_stat_rows)
    write_csv(
        paths["section_top"],
        [
            {"section_title": title, "chunk_count": int(count)}
            for title, count in stats["section_counter"].most_common(80)
        ],
    )
    quality_rows = [
        {"metric": "chunk_id_global_unique", "value": int(stats["duplicate_chunk_ids"]) == 0},
        {"metric": "doc_id_missing_count", "value": int(stats["doc_id_missing"])},
        {"metric": "chunk_index_non_continuous_doc_count", "value": int(stats["non_continuous_docs"])},
        {"metric": "total_chunks_mismatch_doc_count", "value": int(stats["total_chunks_mismatch_docs"])},
        {"metric": "token_count_over_512_count", "value": int(stats["token_over_512"])},
        {"metric": "empty_text_chunk_count", "value": int(stats["empty_text_chunks"])},
        {"metric": "short_chunk_count_token_lt_20", "value": int(stats["short_chunks"])},
        {"metric": "garbled_chunk_count", "value": int(stats["garbled_chunks"])},
        {"metric": "source_title_missing_count", "value": int(stats["source_title_missing"])},
        {"metric": "pmcid_missing_count", "value": int(stats["pmcid_missing"])},
        {"metric": "source_file_missing_count", "value": int(stats["source_file_missing"])},
        {"metric": "semantic_section_without_section_title", "value": int(stats["semantic_without_section_title"])},
        {"metric": "recursive_fallback_configured_overlap", "value": int(args.chunk_overlap)},
        {"metric": "recursive_fallback_multi_chunk_doc_count", "value": int(stats["recursive_multi_doc_count"])},
    ]
    for decision, count in sorted(stats["quality_counts"].items()):
        quality_rows.append({"metric": f"quality_decision_{decision}", "value": int(count)})
    write_csv(paths["quality_flags"], quality_rows)

    processing_stats = {
        "processed_date": pd.Timestamp.now().isoformat(),
        "data_split": args.output_prefix,
        "original_documents": total_docs,
        "processed_documents": processed_documents,
        "chunked_documents": chunked_documents,
        "excluded_documents": excluded_count,
        "total_chunks": total_chunks,
        "chunks_per_doc": stats_for(chunks_per_doc, "chunks_per_doc"),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "whole_doc_token_limit": args.whole_doc_token_limit,
        "tokenizer": MODEL,
        "tokenizer_ref": str(tokenizer_ref),
        "output_manifest": str(paths["manifest"]),
        "chroma_created": False,
        "embedding_created": False,
    }
    paths["processing_stats"].write_text(json.dumps(processing_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(paths["manifest"], manifest_rows)

    if not args.no_jsonl_sample:
        write_sample_jsonl(part_paths, paths["preview_jsonl"], SAMPLE_LIMIT)
    samples = collect_preview_samples(part_paths, paths["excluded"])
    write_preview_markdown(paths["preview_md"], samples, args.output_prefix)
    write_quality_report(paths["report"], args, stats, token_stat_rows, route_rows, manifest_rows)


def write_quality_report(
    path: Path,
    args,
    stats: dict[str, object],
    token_stat_rows: list[dict[str, object]],
    route_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
) -> None:
    chunk_tokens = token_stat_rows[1]
    chunks_per_doc = token_stat_rows[0]
    route_table = markdown_table_from_rows(route_rows) if route_rows else ""
    manifest_table = markdown_table_from_rows(manifest_rows, max_rows=12) if manifest_rows else ""
    excluded_reasons = stats["excluded_reason_counts"]
    excluded_lines = "\n".join(f"- `{k}`: `{v}`" for k, v in sorted(excluded_reasons.items())) or "- 无 excluded 文献"
    body = f"""# 文档解析与分割质量验证报告

## 1. 任务目标与边界

本轮只生成全文级 chunk 数据集，不生成摘要级 chunk，不做 embedding，不做 Chroma 入库，不做 RAG 问答，也不生成 PDF。

## 2. 数据来源与处理规模

数据来自 PMC OA `oa_comm/xml` 本地 XML。当前运行前缀为 `{args.output_prefix}`，输入 XML 数为 `{args.selected_xml_count}`。

## 3. 清洗规则

- `body` 缺失的文献不进入全文 chunk dataset，写入 `excluded_documents`，原因为 `no_body_for_fulltext`。
- `abstract` 缺失不作为丢弃条件，因为本轮主文本来自全文 `body`。
- `title` 缺失不丢弃，chunk text 中使用 `[Missing title: doc_id]`，metadata 中保留 `title_missing=true`。
- `doc_id` 优先使用 PMCID，因为 PMCID 在上周统计中 100% 可用；PMID 保留为追溯 metadata。
- `title`、`abstract`、`body` 均缺失时写入 excluded，原因为 `drop_no_text`。
- `excluded_documents` 表用于保留未入库文献、原因和可用 fallback 信息，避免静默跳过。

## 4. 分割策略

- `whole_document_under_512`：`title + body` 不超过 `{args.whole_doc_token_limit}` tokens，整体生成 1 个 chunk。
- `semantic_section`：长正文且 XML 有可用 section title，先按 section 切分，超长 section 内再递归切分。
- `recursive_fallback_no_section`：长正文但无可用 section title，对 `title + body` 使用重叠递归切分。
- `no_body_for_fulltext`：无正文，不生成全文 chunk，仅记录 excluded。

## 5. 输出数据结构

每个 chunk 包含 `chunk_id`、`text`、`doc_id`、`chunk_index`、`total_chunks`、`source_title`、`token_count`、`source_file`、`journal`、`pub_date`、`pub_year`、`pmid`、`pmcid`、`article_type`、`section_title`、`section_title_norm`、`split_strategy`、`quality_decision`、`title_missing`、`body_missing`、`chunk_char_len`、`section_index`、`section_chunk_index`。

## 6. 处理统计

- 生成 chunk 文献数：`{stats['chunked_documents']}`
- excluded 文献数：`{stats['excluded_count']}`
- 总 chunk 数：`{stats['total_chunks']}`
- chunks/doc p95：`{chunks_per_doc.get('p95', 0)}`
- chunk token p95：`{chunk_tokens.get('p95', 0)}`
- token_count > 512：`{stats['token_over_512']}`，比例 `{(0 if stats['total_chunks'] == 0 else stats['token_over_512'] / stats['total_chunks']):.2%}`

### split_strategy 分布

{route_table}

### parquet part 摘要

{manifest_table}

## 7. 质量验证结果

- chunk_id 全局唯一：`{stats['duplicate_chunk_ids'] == 0}`
- doc_id 为空数：`{stats['doc_id_missing']}`
- chunk_index 非连续文献数：`{stats['non_continuous_docs']}`
- total_chunks 不匹配文献数：`{stats['total_chunks_mismatch_docs']}`
- chunk_index 最大值不等于 total_chunks - 1 的文献数：`{stats['max_index_mismatch_docs']}`
- chunk token_count 超过 512 数：`{stats['token_over_512']}`
- 空 chunk 数：`{stats['empty_text_chunks']}`
- 极短 chunk 数（token_count < 20）：`{stats['short_chunks']}`
- 乱码 chunk 数：`{stats['garbled_chunks']}`
- source_title 缺失 chunk 数：`{stats['source_title_missing']}`
- pmcid 缺失 chunk 数：`{stats['pmcid_missing']}`
- source_file 缺失 chunk 数：`{stats['source_file_missing']}`
- semantic_section 缺失 section_title 数：`{stats['semantic_without_section_title']}`
- recursive_fallback overlap 设置：`{args.chunk_overlap}` tokens

## 8. 多块文献抽样检查

连续 chunks 已写入 `reports/samples/chunk_preview_{args.output_prefix}.md`。检查重点包括断句是否自然、recursive fallback 是否配置 overlap、同一 section 切出的 chunks 是否保留相同 `section_title`，以及 chunk text 是否包含 Title/Section 上下文。

## 9. excluded 文献检查

未进入全文 chunk dataset 的文献共 `{stats['excluded_count']}` 篇，主要原因如下：

{excluded_lines}

`body` 缺失文献因为本轮只做全文级 chunk，所以不生成 chunk，但已记录到 excluded 表。

## 10. 结论与下一步

本轮输出可作为下一阶段 embedding 输入。下一阶段再按 manifest 分片读取 Parquet，执行 embedding 和 Chroma 入库。
"""
    path.write_text(body, encoding="utf-8")


def process_batch(
    batch_files: list[Path],
    source_root: Path,
    part_id: int,
    total_parts: int,
    global_start_index: int,
    total_docs: int,
    run_started: float,
    args,
    tokenizer,
    splitter,
    pmcid_counter: Counter[str],
    pmid_counter: Counter[str],
    seen_pmcids: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    chunk_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    part_started = time.perf_counter()
    for local_index, path in enumerate(batch_files, start=1):
        try:
            doc = extract_base_document(path, source_root, tokenizer)
        except Exception as exc:
            source_file = str(path.relative_to(source_root))
            fallback_doc = {
                "doc_id": f"SOURCE:{short_hash(source_file)}",
                "pmid": "",
                "pmcid": "",
                "title": "",
                "source_file": source_file,
                "abstract": "",
                "body": "",
                "title_token_len": 0,
                "abstract_token_len": 0,
                "body_token_len": 0,
            }
            excluded_rows.append(excluded_row(fallback_doc, "xml_parse_error", f"{type(exc).__name__}: {exc}"))
            continue

        title = str(doc["title"])
        abstract = str(doc["abstract"])
        body = str(doc["body"])
        pmcid = str(doc["pmcid"])
        if pmcid and pmcid in seen_pmcids:
            excluded_rows.append(excluded_row(doc, "duplicate_pmcid", "duplicate PMCID; first record retained"))
            continue
        if pmcid:
            seen_pmcids.add(pmcid)
        if not title.strip() and not abstract.strip() and not body.strip():
            excluded_rows.append(excluded_row(doc, "drop_no_text"))
            continue
        if not body.strip():
            excluded_rows.append(excluded_row(doc, "no_body_for_fulltext"))
            continue
        encoding_note = encoding_issue_note(title, abstract, body)
        if encoding_note.startswith("severe_encoding_issue") and int(doc["body_token_len"]) <= 0:
            excluded_rows.append(excluded_row(doc, "severe_encoding_issue", encoding_note))
            continue

        chunks, chunk_note = build_chunks_for_doc(
            doc,
            splitter,
            tokenizer,
            args.chunk_size,
            args.chunk_overlap,
            args.whole_doc_token_limit,
            pmid_counter,
        )
        if not chunks:
            excluded_rows.append(excluded_row(doc, "empty_chunk_after_split", chunk_note))
            continue
        if str(doc["pmid"]) and pmid_counter.get(str(doc["pmid"]), 0) > 1:
            for chunk in chunks:
                chunk["quality_decision"] = "need_review"
        chunk_rows.extend(chunks)
        if args.progress_interval > 0 and (
            local_index == 1
            or local_index == len(batch_files)
            or local_index % args.progress_interval == 0
        ):
            overall_done = global_start_index + local_index
            print(
                " | ".join(
                    [
                        progress_line(f"Part {part_id + 1}/{total_parts}", local_index, len(batch_files), part_started),
                        progress_line("Overall", overall_done, total_docs, run_started),
                        f"chunks={len(chunk_rows)} excluded={len(excluded_rows)}",
                    ]
                ),
                flush=True,
            )
    return chunk_rows, excluded_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/pmc_oa_comm")
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--whole_doc_token_limit", type=int, default=DEFAULT_WHOLE_DOC_TOKEN_LIMIT)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_jsonl_sample", action="store_true")
    parser.add_argument("--progress_interval", type=int, default=100)
    args = parser.parse_args()

    started = time.perf_counter()
    project_dir = Path.cwd()
    ensure_project_hf_cache(project_dir)
    ensure_output_dirs()
    if args.force:
        clear_prefix_outputs(args.output_prefix)

    paths = prefix_paths(args.output_prefix)
    log_path = Path(f"logs/11_chunk_oa_comm_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        data_dir = Path(args.data_dir)
        xml_files = sorted(data_dir.rglob("*.xml"))
        if args.max_docs > 0:
            xml_files = xml_files[: args.max_docs]
        args.selected_xml_count = len(xml_files)
        if not xml_files:
            raise RuntimeError(f"No XML files found under {data_dir}")

        print(f"START {datetime.now().isoformat(timespec='seconds')}")
        print(f"Data dir: {data_dir}")
        print(f"XML files: {len(xml_files)}")
        print(f"Output prefix: {args.output_prefix}")
        print(f"HF_HOME: {os.environ.get('HF_HOME', '')}")
        tokenizer_ref = resolve_local_tokenizer(project_dir)
        print(f"Loading tokenizer from: {tokenizer_ref}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_ref), local_files_only=True)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            length_function=lambda text: token_len(tokenizer, text),
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

        print("Scanning PMCID/PMID duplicates...")
        pmcid_counter, pmid_counter = scan_duplicate_ids(xml_files)
        print(
            "Duplicate IDs:",
            {
                "duplicate_pmcid_values": sum(1 for v in pmcid_counter.values() if v > 1),
                "duplicate_pmid_values": sum(1 for v in pmid_counter.values() if v > 1),
            },
        )

        seen_pmcids: set[str] = set()
        manifest_rows: list[dict[str, object]] = []
        total_parts = math.ceil(len(xml_files) / args.batch_size)
        for part_id, start in enumerate(range(0, len(xml_files), args.batch_size)):
            end = min(start + args.batch_size, len(xml_files))
            batch_files = xml_files[start:end]
            output_path = part_file(args.output_prefix, part_id)
            created_at = pd.Timestamp.now().isoformat()
            if output_path.exists() and not args.force:
                df = pd.read_parquet(output_path, columns=["doc_id", "chunk_id", "pmcid"])
                seen_pmcids.update(str(v) for v in df["pmcid"].dropna().unique() if str(v).strip())
                manifest_rows.append(
                    {
                        "part_id": part_id,
                        "part_file": str(output_path),
                        "document_count": int(df["doc_id"].nunique()),
                        "chunk_count": int(df.shape[0]),
                        "file_size_mb": round(output_path.stat().st_size / 1024**2, 3),
                        "start_doc_index": start,
                        "end_doc_index": end - 1,
                        "start_doc_id": str(df["doc_id"].iloc[0]) if not df.empty else "",
                        "end_doc_id": str(df["doc_id"].iloc[-1]) if not df.empty else "",
                        "created_at": created_at,
                    }
                )
                write_csv(paths["manifest"], manifest_rows)
                print(f"Skip existing part {part_id + 1}/{total_parts}: {output_path}")
                continue

            print(f"Processing part {part_id + 1}/{total_parts}: docs {start}-{end - 1}")
            chunk_rows, excluded_rows = process_batch(
                batch_files,
                data_dir,
                part_id,
                total_parts,
                start,
                len(xml_files),
                started,
                args,
                tokenizer,
                splitter,
                pmcid_counter,
                pmid_counter,
                seen_pmcids,
            )
            if chunk_rows:
                pd.DataFrame(chunk_rows, columns=CHUNK_COLUMNS).to_parquet(output_path, index=False)
            else:
                pd.DataFrame(columns=CHUNK_COLUMNS).to_parquet(output_path, index=False)
            append_csv(paths["excluded"], excluded_rows, EXCLUDED_COLUMNS)
            start_doc_id = chunk_rows[0]["doc_id"] if chunk_rows else ""
            end_doc_id = chunk_rows[-1]["doc_id"] if chunk_rows else ""
            manifest_rows.append(
                {
                    "part_id": part_id,
                    "part_file": str(output_path),
                    "document_count": len(batch_files),
                    "chunk_count": len(chunk_rows),
                    "file_size_mb": round(output_path.stat().st_size / 1024**2, 3),
                    "start_doc_index": start,
                    "end_doc_index": end - 1,
                    "start_doc_id": start_doc_id,
                    "end_doc_id": end_doc_id,
                    "created_at": created_at,
                }
            )
            write_csv(paths["manifest"], manifest_rows)
            elapsed = time.perf_counter() - started
            print(
                f"Wrote {output_path} chunks={len(chunk_rows)} excluded={len(excluded_rows)} elapsed={elapsed:.1f}s"
            )

        part_paths = [part_file(args.output_prefix, row["part_id"]) for row in manifest_rows]
        stats = validation_summary(part_paths, paths["excluded"])
        elapsed_seconds = time.perf_counter() - started
        write_tables_and_reports(args, paths, manifest_rows, part_paths, stats, elapsed_seconds, tokenizer_ref)
        print("DONE")
        print(f"Input XML: {len(xml_files)}")
        print(f"Chunked documents: {stats['chunked_documents']}")
        print(f"Excluded documents: {stats['excluded_count']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Chunk token p95: {stats_for(stats['token_counts'], 'chunk_token_count')['p95']}")
        print(f"Token >512 chunks: {stats['token_over_512']}")
        print(f"Duplicate chunk IDs: {stats['duplicate_chunk_ids']}")
        print(f"Manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
