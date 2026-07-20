from __future__ import annotations

import json
import math
import re
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from bs4 import BeautifulSoup

OUTPUT_SUBDIRS = [
    "artifacts/datasets/records",
    "artifacts/metrics/t001_environment",
    "artifacts/metrics/t002_corpus_analysis",
    "artifacts/metrics/t005_routed_minilm",
    "reports/figures",
    "reports/samples",
    "logs",
    "reports/technical",
    "archive/experiments/indexes/chroma_limit500",
]

FIELD_COLUMNS = [
    "record_id",
    "source_file",
    "title",
    "abstract",
    "body",
    "journal",
    "pub_date",
    "pub_year",
    "pmid",
    "pmcid",
    "article_type",
    "text_title_abstract",
    "text_full",
]

TARGET_FIELDS = [
    "title",
    "abstract",
    "body",
    "journal",
    "pub_date",
    "pub_year",
    "pmid",
    "pmcid",
    "article_type",
]

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "has", "have", "had",
    "not", "but", "can", "may", "been", "their", "these", "those", "into", "using", "used", "between",
    "among", "also", "than", "then", "there", "such", "which", "when", "where", "during", "after", "before",
    "our", "all", "one", "two", "more", "most", "other", "over", "under", "within", "without", "each",
    "we", "they", "it", "its", "as", "in", "on", "of", "to", "a", "an", "by", "or", "is", "be", "at",
    "study", "studies", "results", "methods", "background", "conclusion", "conclusions", "objective", "objectives",
}

STRUCTURED_MARKERS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS", "INTRODUCTION", "DISCUSSION"]


def ensure_output_dirs(base: Path = Path(".")) -> None:
    for rel in OUTPUT_SUBDIRS:
        (base / rel).mkdir(parents=True, exist_ok=True)


def text_or_empty(node) -> str:
    if node is None:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def first_text(soup: BeautifulSoup, name: str, attrs: Optional[Dict[str, str]] = None) -> str:
    node = soup.find(name, attrs=attrs or {})
    return text_or_empty(node)


def extract_pub_date(soup: BeautifulSoup) -> Tuple[str, str]:
    candidates = []
    for pub_type in ["epub", "ppub", "collection", None]:
        attrs = {"pub-type": pub_type} if pub_type else {}
        node = soup.find("pub-date", attrs=attrs)
        if node:
            candidates.append(node)
    if not candidates:
        candidates = soup.find_all("pub-date")
    for node in candidates:
        year = first_text(node, "year")
        month = first_text(node, "month")
        day = first_text(node, "day")
        if year:
            pieces = [year]
            if month:
                pieces.append(month.zfill(2) if month.isdigit() else month)
            if day:
                pieces.append(day.zfill(2) if day.isdigit() else day)
            return "-".join(pieces), year
    year = first_text(soup, "year")
    return (year, year) if year else ("", "")


def extract_article_id(soup: BeautifulSoup, id_type: str) -> str:
    node = soup.find("article-id", attrs={"pub-id-type": id_type})
    value = text_or_empty(node)
    if value:
        return value
    if id_type == "pmc":
        node = soup.find("article-id", attrs={"pub-id-type": "pmcid"})
        value = text_or_empty(node)
        if value:
            return value.replace("PMC", "")
    return ""


def parse_pmc_xml(path: Path, record_id: str, source_root: Optional[Path] = None) -> Dict[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml-xml")
    article = soup.find("article")
    article_type = article.get("article-type", "") if article else ""
    title = first_text(soup, "article-title")
    abstract = first_text(soup, "abstract")
    body = first_text(soup, "body")
    journal = first_text(soup, "journal-title") or first_text(soup, "journal-id")
    pub_date, pub_year = extract_pub_date(soup)
    pmid = extract_article_id(soup, "pmid")
    pmcid = extract_article_id(soup, "pmc")
    source_file = str(path.relative_to(source_root)) if source_root else str(path)
    text_title_abstract = join_text([title, abstract])
    text_full = join_text([title, abstract, body])
    return {
        "record_id": record_id,
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
        "text_title_abstract": text_title_abstract,
        "text_full": text_full,
    }


def join_text(parts: Sequence[str]) -> str:
    return "\n\n".join([p.strip() for p in parts if isinstance(p, str) and p.strip()])


def is_empty_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def non_empty(series: pd.Series) -> pd.Series:
    return ~series.apply(is_empty_value)


def write_jsonl(records: Sequence[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_records_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def write_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def setup_tee(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()
        def flush(self):
            for stream in self.streams:
                stream.flush()
    fh = log_path.open("w", encoding="utf-8")
    return fh, redirect_stdout(Tee(sys.stdout, fh)), redirect_stderr(Tee(sys.stderr, fh))


def token_chunk_estimate(length: int, chunk_size: int = 400, chunk_overlap: int = 80) -> int:
    if length <= 0:
        return 0
    if length <= chunk_size:
        return 1
    step = max(1, chunk_size - chunk_overlap)
    return 1 + math.ceil((length - chunk_size) / step)


def simple_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", text or ""))


def safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def disk_usage_row(path: Path) -> Dict[str, str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": f"{usage.total / 1024**3:.2f}",
        "used_gb": f"{usage.used / 1024**3:.2f}",
        "free_gb": f"{usage.free / 1024**3:.2f}",
    }
