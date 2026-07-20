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
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/medical_rag_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from medical_rag.common.pmc import (
    STRUCTURED_MARKERS,
    TARGET_FIELDS,
    disk_usage_row,
    ensure_output_dirs,
    extract_article_id,
    extract_pub_date,
    first_text,
    safe_rate,
    simple_word_count,
    text_or_empty,
    token_chunk_estimate,
    write_markdown,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PREFIX = "limit153121"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
WHOLE_DOC_TOKEN_LIMIT = 512

WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z\-]{2,}\b")
ABBR_RE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,12}\b")
SECTION_PATTERNS = {
    "introduction": re.compile(r"\b(introduction|background)\b", re.I),
    "methods": re.compile(r"\b(methods?|materials and methods|methodology|patients and methods)\b", re.I),
    "results": re.compile(r"\b(results?)\b", re.I),
    "discussion": re.compile(r"\bdiscussion\b", re.I),
    "conclusion": re.compile(r"\b(conclusions?|summary)\b", re.I),
}
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "were",
    "was",
    "are",
    "has",
    "have",
    "had",
    "not",
    "but",
    "can",
    "may",
    "been",
    "their",
    "these",
    "those",
    "into",
    "using",
    "used",
    "between",
    "among",
    "also",
    "than",
    "then",
    "there",
    "such",
    "which",
    "when",
    "where",
    "during",
    "after",
    "before",
    "our",
    "all",
    "one",
    "two",
    "more",
    "most",
    "other",
    "over",
    "under",
    "within",
    "without",
    "each",
    "we",
    "they",
    "its",
    "study",
    "studies",
    "results",
    "methods",
    "background",
    "conclusion",
    "conclusions",
    "objective",
    "objectives",
    "article",
    "articles",
    "figure",
    "figures",
    "table",
    "tables",
    "supplementary",
    "additional",
    "file",
    "files",
    "copyright",
    "license",
    "published",
    "journal",
    "author",
    "authors",
    "contribution",
    "contributions",
    "competing",
    "interest",
    "interests",
    "available",
    "availability",
    "result",
    "method",
    "discussion",
    "introduction",
    "however",
    "shown",
    "number",
    "including",
    "respectively",
    "therefore",
    "although",
    "reported",
    "observed",
}
NOISY_ABBR = {
    "PDF",
    "XML",
    "HTML",
    "FIG",
    "TABLE",
    "REF",
    "SUPPL",
    "PMC",
    "PMID",
    "DOI",
    "BMC",
    "PLOS",
    "USA",
    "UK",
    "WHO",
    "CDC",
}
CONCEPT_VARIANTS = {
    "HIV/AIDS": [
        r"\bHIV(?:-1)?\b",
        r"\bAIDS\b",
        r"\bhuman immunodeficiency virus(?: type 1)?\b",
        r"\bacquired immunodeficiency syndrome\b",
    ],
    "PCR": [
        r"\bPCR\b",
        r"\bpolymerase chain reaction\b",
        r"\breverse transcription PCR\b",
        r"\bRT-PCR\b",
    ],
    "confidence interval": [
        r"\bCI\b",
        r"\bconfidence intervals?\b",
        r"\b95%\s*CI\b",
    ],
    "odds ratio": [
        r"\bOR\b",
        r"\bodds ratios?\b",
    ],
    "cerebrospinal fluid": [
        r"\bCSF\b",
        r"\bcerebrospinal fluid\b",
    ],
    "myocardial infarction": [
        r"\bMI\b",
        r"\bmyocardial infarction\b",
        r"\bheart attack\b",
    ],
    "Pneumocystis pneumonia": [
        r"\bPCP\b",
        r"\bPneumocystis pneumonia\b",
        r"\bPneumocystis jiroveci\b",
        r"\bPneumocystis carinii\b",
    ],
    "non-small cell lung cancer": [
        r"\bNSCLC\b",
        r"\bnon-small cell lung cancer\b",
        r"\bnon-small-cell lung cancer\b",
    ],
    "ulcerative colitis": [
        r"\bUC\b",
        r"\bulcerative colitis\b",
        r"\binflammatory bowel disease\b",
        r"\bIBD\b",
    ],
}
CASE_SENSITIVE_PATTERNS = {r"\bHIV(?:-1)?\b", r"\bAIDS\b", r"\bPCR\b", r"\bRT-PCR\b", r"\bCI\b", r"\bOR\b"}


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


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def filtered_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) >= 3]


def update_ngram_counter(counter: Counter[tuple[str, ...]], tokens: list[str], n: int) -> None:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        if any(t in STOPWORDS for t in gram):
            continue
        counter[gram] += 1


def top_ngrams(counter: Counter[tuple[str, ...]], limit: int = 80) -> list[dict[str, object]]:
    return [{"term": " ".join(k), "count": v} for k, v in counter.most_common(limit)]


def update_abbreviations(counter: Counter[str], text: str) -> None:
    counter.update(a for a in ABBR_RE.findall(text or "") if a not in NOISY_ABBR and not a.isdigit())


def count_any(patterns: list[str], text: str) -> int:
    total = 0
    for pattern in patterns:
        flags = 0 if pattern in CASE_SENSITIVE_PATTERNS else re.I
        total += len(re.findall(pattern, text or "", flags=flags))
    return total


def text_hash(text: str) -> str:
    cleaned = " ".join((text or "").lower().split())
    if not cleaned:
        return ""
    return hashlib.sha1(cleaned.encode("utf-8", errors="ignore")).hexdigest()


def xml_section_titles(body) -> list[str]:
    if body is None:
        return []
    titles: list[str] = []
    for sec in body.find_all("sec"):
        title = sec.find("title", recursive=False)
        if title is not None:
            text = text_or_empty(title)
            if text:
                titles.append(text)
    return titles


def marker_flags(titles: list[str]) -> dict[str, bool]:
    joined = "\n".join(titles)
    return {name: bool(pattern.search(joined)) for name, pattern in SECTION_PATTERNS.items()}


def token_len(tokenizer, text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def preview_text(text: str, max_chars: int = 1300) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def route_name(full_token_len: int, has_sections: bool) -> str:
    if full_token_len <= WHOLE_DOC_TOKEN_LIMIT:
        return "whole_document_under_512"
    if has_sections:
        return "semantic_section"
    return "recursive_fallback_no_section"


def strategy_label(route: str) -> str:
    if route == "whole_document_under_512":
        return "整体不分割"
    if route == "semantic_section":
        return "按语义章节分割"
    return "重叠滑动窗口"


def choose_doc_id(pmid: str, pmcid: str, fallback: str) -> str:
    if pmid:
        return f"PMID:{pmid}"
    if pmcid:
        clean = pmcid if str(pmcid).startswith("PMC") else f"PMC{pmcid}"
        return clean
    return fallback


def parse_light_record(path: Path, record_id: str, source_root: Path, tokenizer) -> tuple[dict, dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml-xml")
    article = soup.find("article")
    body_node = soup.find("body")
    article_type = article.get("article-type", "") if article else ""
    title = first_text(soup, "article-title")
    abstract = first_text(soup, "abstract")
    body = text_or_empty(body_node)
    journal = first_text(soup, "journal-title") or first_text(soup, "journal-id")
    pub_date, pub_year = extract_pub_date(soup)
    pmid = extract_article_id(soup, "pmid")
    pmcid = extract_article_id(soup, "pmc")
    source_file = str(path.relative_to(source_root))
    text_title_abstract = "\n\n".join(part for part in [title.strip(), abstract.strip()] if part)
    text_full = "\n\n".join(part for part in [title.strip(), abstract.strip(), body.strip()] if part)
    title_abs_len = token_len(tokenizer, text_title_abstract)
    full_len = token_len(tokenizer, text_full)
    titles = xml_section_titles(body_node)
    flags = marker_flags(titles)
    has_sections = bool(titles)
    route = route_name(full_len, has_sections)
    too_short_abstract = len(abstract.strip()) < 50 or simple_word_count(abstract) < 20
    empty_title = not title.strip()
    empty_abstract = not abstract.strip()
    empty_body = not body.strip()
    encoding_issue = "�" in f"{title} {abstract} {body}"
    if empty_title and empty_abstract and empty_body:
        quality_decision = "drop_no_text"
    elif encoding_issue or (empty_abstract and empty_body):
        quality_decision = "need_review"
    elif too_short_abstract or empty_abstract or empty_body or empty_title:
        quality_decision = "keep_with_warning"
    else:
        quality_decision = "keep"
    row = {
        "record_id": record_id,
        "doc_id": choose_doc_id(pmid, pmcid, record_id),
        "source_file": source_file,
        "title": title,
        "journal": journal,
        "pub_date": pub_date,
        "pub_year": pub_year,
        "pmid": pmid,
        "pmcid": pmcid,
        "article_type": article_type,
        "title_missing": empty_title,
        "abstract_missing": empty_abstract,
        "body_missing": empty_body,
        "title_abstract_token_len": title_abs_len,
        "full_token_len": full_len,
        "exceeds_512_title_abstract": title_abs_len > WHOLE_DOC_TOKEN_LIMIT,
        "exceeds_512_full": full_len > WHOLE_DOC_TOKEN_LIMIT,
        "estimated_chunks_title_abstract": token_chunk_estimate(title_abs_len, CHUNK_SIZE, CHUNK_OVERLAP),
        "estimated_chunks_full": token_chunk_estimate(full_len, CHUNK_SIZE, CHUNK_OVERLAP),
        "section_title_count": len(titles),
        "has_any_section_title": has_sections,
        "has_introduction": flags["introduction"],
        "has_methods": flags["methods"],
        "has_results": flags["results"],
        "has_discussion": flags["discussion"],
        "has_conclusion": flags["conclusion"],
        "imrad_core": flags["introduction"] and flags["methods"] and flags["results"] and flags["discussion"],
        "imrad_with_conclusion": flags["introduction"]
        and flags["methods"]
        and flags["results"]
        and flags["discussion"]
        and flags["conclusion"],
        "quality_decision": quality_decision,
        "recommended_split_strategy": route,
        "recommended_split_strategy_cn": strategy_label(route),
    }
    scratch = {
        "abstract": abstract,
        "body": body,
        "text_full": text_full,
        "text_title_abstract": text_title_abstract,
        "section_titles": titles,
        "title_abstract_hash": text_hash(text_title_abstract),
    }
    return row, scratch


def save_hist(series: list[int], path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=50, color="#4C78A8", edgecolor="white")
    plt.title(title)
    plt.xlabel("Token length")
    plt.ylabel("Document count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_box(series: list[int], path: Path, title: str) -> None:
    plt.figure(figsize=(8, 2.8))
    plt.boxplot(series, vert=False)
    plt.title(title)
    plt.xlabel("Token length")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def stats_for(values: list[int], label: str) -> dict[str, object]:
    s = pd.Series(values, dtype="int64")
    return {
        "text_field": label,
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": int(s.min()) if len(s) else 0,
        "max": int(s.max()) if len(s) else 0,
        "p50": float(s.quantile(0.50)) if len(s) else 0,
        "p75": float(s.quantile(0.75)) if len(s) else 0,
        "p90": float(s.quantile(0.90)) if len(s) else 0,
        "p95": float(s.quantile(0.95)) if len(s) else 0,
        "p99": float(s.quantile(0.99)) if len(s) else 0,
        "over_512_count": int((s > 512).sum()),
        "over_512_rate": float((s > 512).mean()) if len(s) else 0,
        "over_1024_count": int((s > 1024).sum()),
        "over_1024_rate": float((s > 1024).mean()) if len(s) else 0,
    }


def write_rows(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    shown = rows[:limit] if limit else rows
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in shown]
    return "\n".join([header, sep, *body])


def rate(count: int, total: int) -> str:
    return f"{safe_rate(count, total):.2%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/pmc_oa_comm")
    parser.add_argument("--output_prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--expected_total", type=int, default=153121)
    parser.add_argument("--sample_per_group", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    project_dir = Path.cwd()
    ensure_project_hf_cache(project_dir)
    ensure_output_dirs()
    Path("reports/formal").mkdir(parents=True, exist_ok=True)
    Path("artifacts/metrics/t006_fullscale_analysis").mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    xml_files = sorted(data_dir.rglob("*.xml"))
    if len(xml_files) != args.expected_total:
        raise RuntimeError(f"Expected {args.expected_total} XML files, found {len(xml_files)}")

    log_path = Path(f"logs/10_analyze_oa_comm_{args.output_prefix}.log")
    light_path = Path(f"artifacts/metrics/t006_fullscale_analysis/pmc_records_light_{args.output_prefix}.csv")
    fieldnames = [
        "record_id",
        "doc_id",
        "source_file",
        "title",
        "journal",
        "pub_date",
        "pub_year",
        "pmid",
        "pmcid",
        "article_type",
        "title_missing",
        "abstract_missing",
        "body_missing",
        "title_abstract_token_len",
        "full_token_len",
        "exceeds_512_title_abstract",
        "exceeds_512_full",
        "estimated_chunks_title_abstract",
        "estimated_chunks_full",
        "section_title_count",
        "has_any_section_title",
        "has_introduction",
        "has_methods",
        "has_results",
        "has_discussion",
        "has_conclusion",
        "imrad_core",
        "imrad_with_conclusion",
        "quality_decision",
        "recommended_split_strategy",
        "recommended_split_strategy_cn",
    ]

    with log_path.open("w", encoding="utf-8") as log, light_path.open("w", encoding="utf-8", newline="") as light_f:
        def log_print(message: str) -> None:
            print(message)
            log.write(message + "\n")
            log.flush()

        log_print(f"START {datetime.now().isoformat(timespec='seconds')}")
        log_print(f"Data dir: {data_dir}")
        log_print(f"XML files: {len(xml_files)}")
        log_print(f"Python: {os.sys.executable}")
        log_print(f"HF_HOME: {os.environ.get('HF_HOME', '')}")
        tokenizer_ref = resolve_local_tokenizer(project_dir)
        log_print(f"Loading tokenizer from: {tokenizer_ref}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_ref), local_files_only=True)

        writer = csv.DictWriter(light_f, fieldnames=fieldnames)
        writer.writeheader()

        failures: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        title_abs_lengths: list[int] = []
        full_lengths: list[int] = []
        missing_counts = Counter()
        quality_counts = Counter()
        metadata_nonempty = Counter()
        route_counts = Counter()
        route_est_chunks = Counter()
        section_title_counter = Counter()
        structured_marker_counter = Counter()
        title_abstract_hashes = Counter()
        pmid_counter = Counter()
        pmcid_counter = Counter()
        unigram_counter: Counter[tuple[str, ...]] = Counter()
        bigram_counter: Counter[tuple[str, ...]] = Counter()
        trigram_counter: Counter[tuple[str, ...]] = Counter()
        abbr_counter: Counter[str] = Counter()
        concept_counter = Counter()

        for idx, path in enumerate(xml_files, start=1):
            record_id = f"pmc_{idx:06d}"
            try:
                row, scratch = parse_light_record(path, record_id, data_dir, tokenizer)
            except Exception as exc:
                failures.append(
                    {
                        "record_id": record_id,
                        "source_file": str(path.relative_to(data_dir)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if idx % 1000 == 0:
                    log_print(f"Processed {idx}/{len(xml_files)}; failures={len(failures)}")
                continue

            writer.writerow(row)
            rows.append(row)
            title_abs_lengths.append(int(row["title_abstract_token_len"]))
            full_lengths.append(int(row["full_token_len"]))
            quality_counts[str(row["quality_decision"])] += 1
            route = str(row["recommended_split_strategy"])
            route_counts[route] += 1
            route_est_chunks[route] += int(row["estimated_chunks_full"])
            for field in TARGET_FIELDS:
                if field == "title":
                    value = row["title"]
                elif field == "journal":
                    value = row["journal"]
                elif field == "pub_date":
                    value = row["pub_date"]
                elif field == "pub_year":
                    value = row["pub_year"]
                elif field == "pmid":
                    value = row["pmid"]
                elif field == "pmcid":
                    value = row["pmcid"]
                elif field == "article_type":
                    value = row["article_type"]
                elif field == "abstract":
                    value = scratch["abstract"]
                elif field == "body":
                    value = scratch["body"]
                else:
                    value = ""
                if str(value).strip():
                    metadata_nonempty[field] += 1
                else:
                    missing_counts[field] += 1
            if row["pmid"]:
                pmid_counter[str(row["pmid"])] += 1
            if row["pmcid"]:
                pmcid_counter[str(row["pmcid"])] += 1
            if scratch["title_abstract_hash"]:
                title_abstract_hashes[scratch["title_abstract_hash"]] += 1
            section_title_counter.update(t.lower() for t in scratch["section_titles"])
            abstract_upper = str(scratch["abstract"]).upper()
            for marker in STRUCTURED_MARKERS:
                if re.search(rf"\b{marker}\b", abstract_upper):
                    structured_marker_counter[marker] += 1
            text_full = str(scratch["text_full"])
            toks = filtered_words(text_full)
            update_ngram_counter(unigram_counter, toks, 1)
            update_ngram_counter(bigram_counter, toks, 2)
            update_ngram_counter(trigram_counter, toks, 3)
            update_abbreviations(abbr_counter, text_full)
            for concept, patterns in CONCEPT_VARIANTS.items():
                concept_counter[concept] += count_any(patterns, text_full)
            if idx % 1000 == 0:
                elapsed = time.perf_counter() - started
                log_print(f"Processed {idx}/{len(xml_files)}; failures={len(failures)}; elapsed={elapsed:.1f}s")

    total = len(rows)
    parsed_failed = len(failures)
    parsed_success = total
    if parsed_success + parsed_failed != len(xml_files):
        raise RuntimeError("Parsed success + failed does not match XML count")

    rows_df = pd.DataFrame(rows)
    duplicate_title_abstract_count = sum(count for count in title_abstract_hashes.values() if count > 1)
    duplicate_pmid_count = sum(count for count in pmid_counter.values() if count > 1)
    duplicate_pmcid_count = sum(count for count in pmcid_counter.values() if count > 1)

    parse_summary_rows = [
        {"metric": "data_dir", "value": str(data_dir)},
        {"metric": "expected_total_xml", "value": args.expected_total},
        {"metric": "selected_xml", "value": len(xml_files)},
        {"metric": "parsed_success", "value": parsed_success},
        {"metric": "parsed_failed", "value": parsed_failed},
        {"metric": "output_prefix", "value": args.output_prefix},
        {"metric": "full_text_saved", "value": "false"},
        {"metric": "chunk_dataset_created", "value": "false"},
        {"metric": "chroma_created", "value": "false"},
    ]
    pd.DataFrame(parse_summary_rows).to_csv(f"artifacts/metrics/t006_fullscale_analysis/parse_summary_{args.output_prefix}.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(f"artifacts/metrics/t006_fullscale_analysis/parse_failures_{args.output_prefix}.csv", index=False)

    missing_rows = []
    for field in TARGET_FIELDS:
        non_empty = int(metadata_nonempty[field])
        missing = parsed_success - non_empty
        missing_rows.append(
            {
                "field": field,
                "total_count": parsed_success,
                "non_empty_count": non_empty,
                "missing_count": missing,
                "missing_rate": safe_rate(missing, parsed_success),
            }
        )
    pd.DataFrame(missing_rows).to_csv(f"artifacts/metrics/t006_fullscale_analysis/missing_rate_{args.output_prefix}.csv", index=False)

    quality_rows = [
        {"metric": "total_records", "value": parsed_success},
        {"metric": "empty_title_count", "value": int(rows_df["title_missing"].sum())},
        {"metric": "empty_abstract_count", "value": int(rows_df["abstract_missing"].sum())},
        {"metric": "empty_body_count", "value": int(rows_df["body_missing"].sum())},
        {"metric": "duplicate_title_abstract_count", "value": int(duplicate_title_abstract_count)},
        {"metric": "duplicate_pmid_count", "value": int(duplicate_pmid_count)},
        {"metric": "duplicate_pmcid_count", "value": int(duplicate_pmcid_count)},
    ]
    for decision, count in sorted(quality_counts.items()):
        quality_rows.append({"metric": f"quality_decision_{decision}", "value": int(count)})
    pd.DataFrame(quality_rows).to_csv(f"artifacts/metrics/t006_fullscale_analysis/quality_summary_{args.output_prefix}.csv", index=False)

    usage = {
        "title": "检索文本增强、结果展示",
        "journal": "期刊 metadata filter",
        "pub_year": "年份 metadata filter",
        "pmid": "PubMed 原文追溯",
        "pmcid": "PMC 原文追溯",
        "source_file": "本地溯源和调试",
    }
    meta_rows = []
    for field, future in usage.items():
        if field == "source_file":
            availability = 1.0
        else:
            availability = safe_rate(int(metadata_nonempty[field]), parsed_success)
        meta_rows.append(
            {
                "field": field,
                "availability_rate": availability,
                "future_rag_usage": future,
                "recommendation": "保留" if availability > 0 else "暂不依赖",
            }
        )
    pd.DataFrame(meta_rows).to_csv(f"artifacts/metrics/t006_fullscale_analysis/metadata_summary_{args.output_prefix}.csv", index=False)

    token_stats = [
        stats_for(title_abs_lengths, "text_title_abstract"),
        stats_for(full_lengths, "text_full"),
    ]
    pd.DataFrame(token_stats).to_csv(f"artifacts/metrics/t006_fullscale_analysis/token_length_stats_{args.output_prefix}.csv", index=False)
    rows_df[
        [
            "record_id",
            "source_file",
            "title_abstract_token_len",
            "full_token_len",
            "exceeds_512_title_abstract",
            "exceeds_512_full",
            "estimated_chunks_title_abstract",
            "estimated_chunks_full",
            "recommended_split_strategy",
        ]
    ].to_csv(f"artifacts/metrics/t006_fullscale_analysis/token_length_records_light_{args.output_prefix}.csv", index=False)

    section_cols = [
        "record_id",
        "source_file",
        "full_token_len",
        "estimated_chunks_full",
        "section_title_count",
        "has_any_section_title",
        "has_introduction",
        "has_methods",
        "has_results",
        "has_discussion",
        "has_conclusion",
        "imrad_core",
        "imrad_with_conclusion",
        "recommended_split_strategy",
    ]
    rows_df[section_cols].to_csv(f"artifacts/metrics/t006_fullscale_analysis/full_text_section_analysis_light_{args.output_prefix}.csv", index=False)
    pd.DataFrame(
        [{"section_title": title, "count": count} for title, count in section_title_counter.most_common(80)]
    ).to_csv(f"artifacts/metrics/t006_fullscale_analysis/full_text_section_title_top80_{args.output_prefix}.csv", index=False)

    split_rows = [
        {"metric": "records", "value": parsed_success},
        {"metric": "chunk_size_for_strategy", "value": CHUNK_SIZE},
        {"metric": "chunk_overlap_for_strategy", "value": CHUNK_OVERLAP},
        {"metric": "whole_doc_token_limit", "value": WHOLE_DOC_TOKEN_LIMIT},
        {"metric": "full_token_mean", "value": float(pd.Series(full_lengths).mean())},
        {"metric": "full_token_median", "value": float(pd.Series(full_lengths).median())},
        {"metric": "full_token_p95", "value": float(pd.Series(full_lengths).quantile(0.95))},
        {"metric": "full_token_p99", "value": float(pd.Series(full_lengths).quantile(0.99))},
        {"metric": "estimated_total_full_chunks_if_implemented", "value": int(rows_df["estimated_chunks_full"].sum())},
        {"metric": "records_with_section_titles", "value": int(rows_df["has_any_section_title"].sum())},
        {"metric": "records_with_section_titles_rate", "value": float(rows_df["has_any_section_title"].mean())},
        {"metric": "records_with_imrad_core", "value": int(rows_df["imrad_core"].sum())},
        {"metric": "records_with_imrad_core_rate", "value": float(rows_df["imrad_core"].mean())},
    ]
    for route, count in sorted(route_counts.items()):
        split_rows.append({"metric": f"route_records_{route}", "value": int(count)})
        split_rows.append({"metric": f"route_estimated_chunks_{route}", "value": int(route_est_chunks[route])})
    pd.DataFrame(split_rows).to_csv(f"artifacts/metrics/t006_fullscale_analysis/full_text_split_strategy_summary_{args.output_prefix}.csv", index=False)

    pd.DataFrame(top_ngrams(unigram_counter)).to_csv(f"artifacts/metrics/t006_fullscale_analysis/fulltext_high_freq_unigrams_{args.output_prefix}.csv", index=False)
    pd.DataFrame(top_ngrams(bigram_counter)).to_csv(f"artifacts/metrics/t006_fullscale_analysis/fulltext_high_freq_bigrams_{args.output_prefix}.csv", index=False)
    pd.DataFrame(top_ngrams(trigram_counter)).to_csv(f"artifacts/metrics/t006_fullscale_analysis/fulltext_high_freq_trigrams_{args.output_prefix}.csv", index=False)
    pd.DataFrame([{"abbreviation": k, "count": v} for k, v in abbr_counter.most_common(80)]).to_csv(
        f"artifacts/metrics/t006_fullscale_analysis/fulltext_abbreviation_top80_{args.output_prefix}.csv", index=False
    )
    pd.DataFrame([{"concept": k, "total_mentions": v} for k, v in concept_counter.most_common()]).to_csv(
        f"artifacts/metrics/t006_fullscale_analysis/fulltext_concept_variants_summary_{args.output_prefix}.csv", index=False
    )
    pd.DataFrame(
        [
            {"marker": marker, "count": int(structured_marker_counter[marker]), "rate": safe_rate(int(structured_marker_counter[marker]), parsed_success)}
            for marker in STRUCTURED_MARKERS
        ]
    ).to_csv(f"artifacts/metrics/t006_fullscale_analysis/structured_abstract_markers_{args.output_prefix}.csv", index=False)

    fig_dir = Path("reports/figures")
    save_hist(title_abs_lengths, fig_dir / f"title_abstract_token_length_hist_{args.output_prefix}.png", "Title + abstract token length")
    save_box(title_abs_lengths, fig_dir / f"title_abstract_token_length_box_{args.output_prefix}.png", "Title + abstract token length")
    save_hist(full_lengths, fig_dir / f"full_text_token_length_hist_{args.output_prefix}.png", "Full text token length")
    save_box(full_lengths, fig_dir / f"full_text_token_length_box_{args.output_prefix}.png", "Full text token length")

    q33 = rows_df["full_token_len"].quantile(1 / 3)
    q66 = rows_df["full_token_len"].quantile(2 / 3)
    def group_name(length: int) -> str:
        if length <= q33:
            return "short"
        if length <= q66:
            return "medium"
        return "long"

    rows_df["full_length_group"] = rows_df["full_token_len"].apply(group_name)
    sample_rows = []
    for group in ["short", "medium", "long"]:
        sample = rows_df[rows_df["full_length_group"] == group].sample(
            n=min(args.sample_per_group, int((rows_df["full_length_group"] == group).sum())),
            random_state=42,
        )
        sample_rows.append(sample)
    sample_df = pd.concat(sample_rows, ignore_index=True)
    sample_df.to_csv(f"artifacts/metrics/t006_fullscale_analysis/fulltext_domain_sample_summary_{args.output_prefix}.csv", index=False)
    sample_lines = [
        "# 15w 全文分层抽样阅读包",
        "",
        f"- 分层依据：`full_token_len` 三分位；short <= {int(q33)}，medium <= {int(q66)}，long > {int(q66)}。",
        f"- 每层抽样：{args.sample_per_group} 篇；共 {len(sample_df)} 篇。",
        "- 本文件只保存轻量预览，不保存全量正文。",
        "",
    ]
    sample_by_source = {row["source_file"]: row for _, row in sample_df.iterrows()}
    for xml_path in xml_files:
        rel = str(xml_path.relative_to(data_dir))
        if rel not in sample_by_source:
            continue
        row = sample_by_source[rel]
        try:
            raw = xml_path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(raw, "lxml-xml")
            body_preview = text_or_empty(soup.find("body"))
            abstract_preview = first_text(soup, "abstract")
            text_preview = preview_text("\n\n".join([str(row["title"]), abstract_preview, body_preview]))
        except Exception as exc:
            text_preview = f"PREVIEW FAILED: {type(exc).__name__}: {exc}"
        sample_lines.extend(
            [
                f"## {row['full_length_group']} | {row['record_id']} | {int(row['full_token_len'])} tokens",
                f"- title: {row['title']}",
                f"- journal/year: {row['journal']} / {row['pub_year']}",
                f"- pmid/pmcid: {row['pmid']} / {row['pmcid']}",
                f"- source_file: {row['source_file']}",
                f"- strategy: {row['recommended_split_strategy_cn']} (`{row['recommended_split_strategy']}`)",
                "",
                text_preview,
                "",
            ]
        )
    sample_md = Path(f"reports/samples/fulltext_stratified_sample_for_review_{args.output_prefix}.md")
    sample_md.write_text("\n".join(sample_lines), encoding="utf-8")

    disk = disk_usage_row(project_dir)
    elapsed = time.perf_counter() - started
    summary_rows = [
        {"metric": "processed_date", "value": datetime.now().isoformat(timespec="seconds")},
        {"metric": "input_xml", "value": len(xml_files)},
        {"metric": "parsed_success", "value": parsed_success},
        {"metric": "parsed_failed", "value": parsed_failed},
        {"metric": "elapsed_seconds", "value": f"{elapsed:.3f}"},
        {"metric": "output_prefix", "value": args.output_prefix},
        {"metric": "light_table", "value": str(light_path)},
        {"metric": "report", "value": f"reports/formal/RAG数据分析与设计说明_{args.output_prefix}.md"},
        {"metric": "chroma_created", "value": "false"},
        {"metric": "chunk_dataset_created", "value": "false"},
        {"metric": "pdf_created", "value": "false"},
        *[{"metric": f"disk_{k}", "value": v} for k, v in disk.items()],
    ]
    pd.DataFrame(summary_rows).to_csv(f"artifacts/metrics/t006_fullscale_analysis/fullscale_analysis_summary_{args.output_prefix}.csv", index=False)

    missing_lookup = {row["field"]: row for row in missing_rows}
    token_lookup = {row["text_field"]: row for row in token_stats}
    ta = token_lookup["text_title_abstract"]
    full = token_lookup["text_full"]
    route_rows = [
        {
            "strategy": strategy_label(route),
            "route": route,
            "records": int(count),
            "rate": rate(int(count), parsed_success),
            "estimated_chunks_if_implemented": int(route_est_chunks[route]),
        }
        for route, count in sorted(route_counts.items())
    ]
    top_abbr = [k for k, _ in abbr_counter.most_common(12)]
    top_uni = [" ".join(k) for k, _ in unigram_counter.most_common(12)]
    top_bi = [" ".join(k) for k, _ in bigram_counter.most_common(10)]

    report_body = f"""
## 1. 任务背景与边界

本报告面向医学 RAG 项目的上周任务：在原 `3028` 篇小样本基础上，将 PMC OA `oa_comm/xml` 数据扩展到 `153121` 篇后，重新完成数据加载、质量评估、长度分析、领域语言理解和文本分割策略制定。

本轮只做分析与策略报告，不生成实际文本块数据集，不做 Chroma 入库，不生成 PDF。下周的“文档解析与分割工作”再根据本文策略生成 chunk 数据集。

## 2. 数据来源与加载

- 数据源：NCBI PMC OA Bulk deprecated `oa_comm/xml`
- 本地目录：`{data_dir}`
- 输入 XML：`{len(xml_files)}`
- 解析成功：`{parsed_success}`
- 解析失败：`{parsed_failed}`
- 轻量全量表：`{light_path}`

本轮轻量全量表不保存 `body` / `text_full`，只保存标题、metadata、长度、质量、章节和策略字段，避免把 15w 篇正文复制成大型 CSV/JSONL。后续实际 chunk 生成会重新读取原始 XML。

## 3. 字段完整性与清洗策略

| 字段 | 非空数量 | 缺失数量 | 缺失率 |
|---|---:|---:|---:|
| title | {int(metadata_nonempty['title'])} | {int(missing_lookup['title']['missing_count'])} | {float(missing_lookup['title']['missing_rate']):.2%} |
| abstract | {int(metadata_nonempty['abstract'])} | {int(missing_lookup['abstract']['missing_count'])} | {float(missing_lookup['abstract']['missing_rate']):.2%} |
| body | {int(metadata_nonempty['body'])} | {int(missing_lookup['body']['missing_count'])} | {float(missing_lookup['body']['missing_rate']):.2%} |
| journal | {int(metadata_nonempty['journal'])} | {int(missing_lookup['journal']['missing_count'])} | {float(missing_lookup['journal']['missing_rate']):.2%} |
| pub_year | {int(metadata_nonempty['pub_year'])} | {int(missing_lookup['pub_year']['missing_count'])} | {float(missing_lookup['pub_year']['missing_rate']):.2%} |
| pmid | {int(metadata_nonempty['pmid'])} | {int(missing_lookup['pmid']['missing_count'])} | {float(missing_lookup['pmid']['missing_rate']):.2%} |
| pmcid | {int(metadata_nonempty['pmcid'])} | {int(missing_lookup['pmcid']['missing_count'])} | {float(missing_lookup['pmcid']['missing_rate']):.2%} |

清洗策略：`title`、`journal`、`pub_year`、`pmcid` 用于展示、过滤和追溯；`abstract` 缺失时不直接丢弃，因为正文通常仍可用；`pmid` 缺失时用 `pmcid/source_file` 兜底。质量标记统计见 `artifacts/metrics/t006_fullscale_analysis/quality_summary_{args.output_prefix}.csv`。

## 4. Metadata 可用性

`journal` 和 `pub_year` 可作为后续检索过滤条件，支持类似“按期刊/年份过滤”的 RAG 查询；`pmid` 和 `pmcid` 用于回答溯源，其中 `pmcid` 是更稳定的 fallback。完整 metadata 表见 `artifacts/metrics/t006_fullscale_analysis/metadata_summary_{args.output_prefix}.csv`。

## 5. Token 长度分布

| 文本 | count | mean | median | p95 | p99 | >512 数 | >512 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| title + abstract | {ta['count']} | {ta['mean']:.2f} | {ta['median']:.2f} | {ta['p95']:.2f} | {ta['p99']:.2f} | {ta['over_512_count']} | {ta['over_512_rate']:.2%} |
| full text | {full['count']} | {full['mean']:.2f} | {full['median']:.2f} | {full['p95']:.2f} | {full['p99']:.2f} | {full['over_512_count']} | {full['over_512_rate']:.2%} |

结论：摘要文本存在长尾，全文几乎都需要切分或章节化处理。长度图见 `reports/figures/*_{args.output_prefix}.png`。

## 6. 领域语言与结构特点

- 任意正文 section title 覆盖率：`{rows_df['has_any_section_title'].mean():.2%}`
- IMRaD core 覆盖率：`{rows_df['imrad_core'].mean():.2%}`
- 含 Conclusion/Summary 的 IMRaD 覆盖率：`{rows_df['imrad_with_conclusion'].mean():.2%}`
- 全文高频缩写：`{", ".join(top_abbr)}`
- 高频 unigram：`{", ".join(top_uni)}`
- 高频 bigram：`{", ".join(top_bi)}`

医学文本存在大量缩写、全称、同义表达和统计符号。后续 prompt/query rewrite 应保留原始术语，同时支持常见缩写与全称互扩展。领域分析表见 `artifacts/metrics/t006_fullscale_analysis/fulltext_high_freq_*_{args.output_prefix}.csv` 和 `artifacts/metrics/t006_fullscale_analysis/fulltext_abbreviation_top80_{args.output_prefix}.csv`。

## 7. 文本分割策略

mentor 给出的三类策略不是全局三选一，而是按文献长度和结构进行条件路由。本轮建议：

{markdown_table(route_rows, ['strategy', 'route', 'records', 'rate', 'estimated_chunks_if_implemented'])}

策略解释：

- **整体不分割**：全文不超过 `{WHOLE_DOC_TOKEN_LIMIT}` tokens 时，保留完整上下文。
- **按语义章节分割**：正文有 XML section title 时，优先按章节保留 Background/Methods/Results/Discussion/Conclusion 等结构；超长章节在下周实际 chunk 阶段再使用 recursive split。
- **重叠滑动窗口兜底**：无明确章节但全文较长时，使用 `RecursiveCharacterTextSplitter`，建议 `chunk_size={CHUNK_SIZE}`、`chunk_overlap={CHUNK_OVERLAP}`。

本轮只输出策略统计，不生成实际 chunk 数据集。下周任务再保存 `chunk_id/text/doc_id/chunk_index/total_chunks/source_title/token_count` 等字段。

## 8. 关键产物

```text
reports/formal/RAG数据分析与设计说明_{args.output_prefix}.md
artifacts/metrics/t006_fullscale_analysis/pmc_records_light_{args.output_prefix}.csv
artifacts/metrics/t006_fullscale_analysis/missing_rate_{args.output_prefix}.csv
artifacts/metrics/t006_fullscale_analysis/quality_summary_{args.output_prefix}.csv
artifacts/metrics/t006_fullscale_analysis/metadata_summary_{args.output_prefix}.csv
artifacts/metrics/t006_fullscale_analysis/token_length_stats_{args.output_prefix}.csv
artifacts/metrics/t006_fullscale_analysis/full_text_split_strategy_summary_{args.output_prefix}.csv
artifacts/metrics/t006_fullscale_analysis/full_text_section_analysis_light_{args.output_prefix}.csv
reports/samples/fulltext_stratified_sample_for_review_{args.output_prefix}.md
```

## 9. 验证结论

- XML 总数验证：`{len(xml_files)}`。
- 解析计数验证：`parsed_success + parsed_failed = {parsed_success + parsed_failed}`。
- 本轮未生成 Chroma、未生成 chunk dataset、未生成 PDF。
- 总耗时：`{elapsed / 60:.2f}` 分钟。
"""
    report_path = Path(f"reports/formal/RAG数据分析与设计说明_{args.output_prefix}.md")
    write_markdown(report_path, "RAG数据分析与设计说明（15w PMC OA）", report_body)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"END {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Elapsed seconds: {elapsed:.3f}\n")
        log.write(f"Wrote report: {report_path}\n")


if __name__ == "__main__":
    main()
