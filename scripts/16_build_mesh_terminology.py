from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class Tee:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal, self.file = sys.stdout, path.open("w", encoding="utf-8")
    def write(self, text: str) -> None:
        self.terminal.write(text); self.file.write(text)
    def flush(self) -> None:
        self.terminal.flush(); self.file.flush()
    def close(self) -> None:
        self.flush(); self.file.close()


def tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def normalize_term(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def is_valid_term(term: str, min_len: int) -> bool:
    return len(term) >= min_len and len(term) <= 256 and not term.isdigit()


def text_at(record: ET.Element, path: str) -> str:
    return (record.findtext(path) or "").strip()


def texts_at(record: ET.Element, path: str) -> list[str]:
    return [item.text.strip() for item in record.findall(path) if item.text and item.text.strip()]


def record_terms(record: ET.Element, preferred: str) -> list[str]:
    values = [preferred]
    values.extend(texts_at(record, "./ConceptList/Concept/ConceptName/String"))
    values.extend(texts_at(record, "./ConceptList/Concept/TermList/Term/String"))
    output, seen = [], set()
    for value in values:
        value = re.sub(r"\s+", " ", value).strip()
        key = normalize_term(value)
        if value and key not in seen:
            output.append(value); seen.add(key)
    return output


def parse_record(record: ET.Element, record_type: str, min_len: int, max_terms: int, skipped: Counter[str]) -> dict[str, Any] | None:
    if record_type == "descriptor":
        raw_id, preferred = text_at(record, "./DescriptorUI"), text_at(record, "./DescriptorName/String")
        trees = texts_at(record, "./TreeNumberList/TreeNumber")
    elif record_type == "supplemental":
        raw_id, preferred = text_at(record, "./SupplementalRecordUI"), text_at(record, "./SupplementalRecordName/String")
        trees = texts_at(record, "./HeadingMappedToList/HeadingMappedTo/DescriptorReferredTo/DescriptorUI")
    else:
        raw_id, preferred = text_at(record, "./QualifierUI"), text_at(record, "./QualifierName/String")
        trees = []
    if not raw_id or not preferred:
        skipped["missing_id_or_preferred"] += 1
        return None
    terms = []
    for term in record_terms(record, preferred):
        if not is_valid_term(term, min_len):
            skipped["invalid_term"] += 1
            continue
        terms.append(term)
    terms = terms[:max_terms]
    if not terms:
        skipped["record_without_valid_terms"] += 1
        return None
    scope_note = text_at(record, "./ConceptList/Concept/ScopeNote") or text_at(record, "./Note")
    return {
        "concept_id": f"MESH:{raw_id}",
        "preferred_term": preferred,
        "terms": terms,
        "sources": ["MeSH"],
        "mesh_record_type": record_type,
        "tree_numbers": trees,
        "scope_note": scope_note,
        "language": "ENG",
        "term_count": len(terms),
    }


def source_files(directory: Path) -> list[tuple[Path, str]]:
    groups = (("desc", "descriptor"), ("supp", "supplemental"), ("qual", "qualifier"))
    found: list[tuple[Path, str]] = []
    for prefix, kind in groups:
        for path in sorted(directory.glob(f"{prefix}*.xml")) + sorted(directory.glob(f"{prefix}*.gz")):
            if path.is_file():
                found.append((path, kind))
    return found


def iter_records(path: Path, kind: str) -> Iterator[ET.Element]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for _, elem in ET.iterparse(handle, events=("end",)):
            if tag_name(elem) == {"descriptor": "DescriptorRecord", "supplemental": "SupplementalRecord", "qualifier": "QualifierRecord"}[kind]:
                yield elem
                elem.clear()


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["concept_id", "preferred_term", "terms_json", "sources_json", "mesh_record_type", "tree_numbers_json", "scope_note", "language", "term_count"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in records:
            writer.writerow({
                "concept_id": item["concept_id"], "preferred_term": item["preferred_term"],
                "terms_json": json.dumps(item["terms"], ensure_ascii=False),
                "sources_json": json.dumps(item["sources"], ensure_ascii=False),
                "mesh_record_type": item["mesh_record_type"],
                "tree_numbers_json": json.dumps(item["tree_numbers"], ensure_ascii=False),
                "scope_note": item["scope_note"], "language": item["language"], "term_count": item["term_count"],
            })


def make_report(stats: dict[str, Any], files: dict[str, Path]) -> str:
    return f"""# MeSH标准术语词典构建报告（mesh）

## 1. 数据来源与范围

标准术语来源为 NLM 当前生产年度 MeSH XML：Descriptor Records 与 Supplemental Concept Records；本轮未接入 UMLS。输入为压缩 XML，采用 xml.etree.ElementTree.iterparse 流式处理，不复制 XML 原文到 outputs。

## 2. 构建方式

每个 Descriptor/SupplementalRecord 聚合 preferred term、ConceptName 与 TermList 为一个同义词组。术语 key 统一为去首尾空白、压缩空白、casefold 后的值；过滤短于最小长度、纯数字和超过 256 字符的噪声词，每个 concept 最多保留指定数量的术语。

## 3. 统计

- descriptor_count: {stats["descriptor_count"]}
- supplemental_count: {stats["supplemental_count"]}
- qualifier_count: {stats["qualifier_count"]}
- concept_count: {stats["concept_count"]}
- total_terms: {stats["total_terms"]}
- unique_terms: {stats["unique_terms"]}
- skipped_terms: {stats["skipped_terms"]}

## 4. 输出

- synonym JSONL: {files["jsonl"]}
- synonym CSV: {files["csv"]}
- term index: {files["index"]}
- stats JSON: {files["stats"]}

## 5. 使用说明与局限

term_to_concept_mesh.json 用于最长匹配和 concept 定位；medical_synonyms_mesh.jsonl 用于从 concept 取得同义词组。MeSH 是英文标准主题词表，中文医学词仍由项目 seed fallback 处理。Supplemental Record 的 tree_numbers 字段保存其 HeadingMappedTo DescriptorUI；这不是 Descriptor Tree Number。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local MeSH terminology and term index from XML or XML.GZ files.")
    parser.add_argument("--mesh_xml_dir", default="data/reference/mesh/2026")
    parser.add_argument("--output_prefix", default="mesh")
    parser.add_argument("--max_terms_per_concept", type=int, default=30)
    parser.add_argument("--min_term_len", type=int, default=2)
    args = parser.parse_args()
    root = project_root()
    input_dir = (root / args.mesh_xml_dir).resolve()
    terminology_dir, tables_dir, reports_dir, logs_dir = (root / "artifacts/terminology/mesh_2026", root / "artifacts/metrics/t010_mesh_query_understanding", root / "reports/formal", root / "logs")
    for directory in (terminology_dir, tables_dir, reports_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"16_build_mesh_terminology_{args.output_prefix}.log"
    tee, old = Tee(log_path), sys.stdout
    sys.stdout = tee
    try:
        files = source_files(input_dir)
        if not files:
            print(f"[ERROR] no desc/supp/qual XML or GZ files found in {input_dir}")
            print("Download or upload official MeSH XML files, for example descYYYY.gz and suppYYYY.gz.")
            return 2
        start, records, skipped, counts = time.time(), [], Counter(), Counter()
        for path, kind in files:
            print(f"[READ] {kind}: {path}")
            for element in iter_records(path, kind):
                counts[f"{kind}_seen"] += 1
                item = parse_record(element, kind, args.min_term_len, args.max_terms_per_concept, skipped)
                if item:
                    records.append(item); counts[f"{kind}_count"] += 1
            print(f"[DONE] {kind}: concepts={counts[f'{kind}_count']}")
        index: dict[str, list[dict[str, Any]]] = {}
        for item in records:
            for term in item["terms"]:
                key = normalize_term(term)
                index.setdefault(key, []).append({
                    "concept_id": item["concept_id"], "preferred_term": item["preferred_term"],
                    "source": "MeSH", "mesh_record_type": item["mesh_record_type"], "tree_numbers": item["tree_numbers"],
                })
        jsonl_path = terminology_dir / f"medical_synonyms_{args.output_prefix}.jsonl"
        csv_path = terminology_dir / f"medical_synonyms_{args.output_prefix}.csv"
        index_path = terminology_dir / f"term_to_concept_{args.output_prefix}.json"
        stats_path = tables_dir / f"medical_terminology_stats_{args.output_prefix}.json"
        report_path = reports_dir / f"MeSH标准术语词典构建报告_{args.output_prefix}.md"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        write_csv(csv_path, records)
        index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        output_sizes = {str(path.relative_to(root)): path.stat().st_size for path in (jsonl_path, csv_path, index_path)}
        stats = {
            "source_files": [str(path.relative_to(root)) for path, _ in files],
            "descriptor_count": counts["descriptor_count"], "supplemental_count": counts["supplemental_count"], "qualifier_count": counts["qualifier_count"],
            "concept_count": len(records), "total_terms": sum(item["term_count"] for item in records), "unique_terms": len(index),
            "skipped_terms": sum(skipped.values()), "skipped_breakdown": dict(skipped),
            "max_terms_per_concept": args.max_terms_per_concept, "min_term_len": args.min_term_len,
            "output_file_sizes_bytes": output_sizes, "elapsed_seconds": round(time.time() - start, 3),
        }
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        paths = {"jsonl": jsonl_path.relative_to(root), "csv": csv_path.relative_to(root), "index": index_path.relative_to(root), "stats": stats_path.relative_to(root)}
        report_path.write_text(make_report(stats, {key: Path(value) for key, value in paths.items()}), encoding="utf-8")
        print(f"[DONE] concepts={stats['concept_count']} total_terms={stats['total_terms']} unique_terms={stats['unique_terms']}")
        print(f"[DONE] index={index_path} report={report_path}")
        return 0
    except (OSError, ET.ParseError) as exc:
        print(f"[ERROR] {exc!r}")
        return 1
    finally:
        sys.stdout = old
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
