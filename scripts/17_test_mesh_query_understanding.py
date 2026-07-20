from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from medical_rag.query.understanding import load_mesh_resources, process_medical_query

TEST_QUERIES = [
    "二甲双胍对心血管疾病有何影响？", "MI treatment", "myocardial infarction treatment",
    "heart attack aspirin mortality", "EGFR mutation lung cancer treatment",
    "HIV reverse transcriptase inhibitor resistance", "type 2 diabetes insulin sensitivity after 2010",
    "PLoS ONE breast cancer gene expression", "SARS coronavirus spike protein", "PCR DNA amplification",
    "warfarin bleeding risk", "2010年关于肺癌EGFR突变的研究",
    "metformin cardiovascular outcome in PLoS ONE", "", "EGFR mutation " * 200,
]


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


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def make_row(raw: str, resources: Any) -> dict[str, str]:
    result = process_medical_query(raw, mesh_resources=resources)
    sources = sorted({entity.source for entity in result.entities})
    status = "invalid" if not result.clean_query else ("warning" if result.warnings else "ok")
    return {
        "raw_query": raw, "clean_query": result.clean_query, "entity_count": str(len(result.entities)),
        "entities_json": dumps([asdict(entity) for entity in result.entities]), "expanded_terms": dumps(result.expanded_terms),
        "vector_query": result.vector_query, "bge_query": result.bge_query, "keyword_query_json": dumps(result.keyword_query),
        "where_filter_json": dumps(result.where_filter), "filter_plan_json": dumps(result.filter_plan),
        "terminology_sources": dumps(sources), "warnings": dumps(result.warnings), "status": status,
    }


def make_report(path: Path, rows: list[dict[str, str]], terminology: Path, index: Path, stats: dict[str, Any]) -> None:
    counts = {name: sum(row["status"] == name for row in rows) for name in ("ok", "warning", "invalid")}
    mesh_cases = sum("MeSH" in row["terminology_sources"] for row in rows)
    text = f"""# MeSH查询理解与增强模块说明（mesh）

## 1. T010 任务目标

T010 使用 MeSH XML 构建本地标准术语词典，并升级 T009 静态规则查询理解。未重新解析 PMC XML、未重建 embedding/Chroma，未调用 OpenAI，也未接入 UMLS。

## 2. 为什么先选 MeSH

MeSH 是 NLM 用于 PubMed/MEDLINE 索引的医学主题词表，与本项目 PMC/PubMed 检索场景直接匹配；其 XML 可公开下载并流式解析。UMLS 作为后续增强，不阻塞本轮。

## 3. XML 来源和解析范围

使用 NLM 2026 Production Year MeSH 的 Descriptor 与 Supplemental Concept XML.GZ。解析 DescriptorUI/Name/TreeNumber、Concept/Term/ScopeNote，以及 SupplementalRecord 与 HeadingMappedTo 信息；Qualifier 未提供时不伪造。

## 4. 词典构建与统计

- concept_count: {stats.get("concept_count")}
- total_terms: {stats.get("total_terms")}
- unique_terms: {stats.get("unique_terms")}
- descriptor_count: {stats.get("descriptor_count")}
- supplemental_count: {stats.get("supplemental_count")}
- terminology: {terminology}
- term_index: {index}

## 5. MeSH 优先查询理解

query_understanding.py 加载 JSONL concept groups 和 term index，英文术语按最长 n-gram 精确匹配，优先输出 concept_id、MeSH source、record type、tree_numbers 和同 concept 同义词。未命中的中文及项目常用词继续走静态 seed fallback，例如二甲双胍、心血管疾病、MI、PCR。

## 6. 清洗、扩展与检索 query

保留 T009 的空白/标点清洗、空 query 安全拦截和超长截断。每个 MeSH concept 最多保留五个短且去重的同义词；vector_query 保留原始核心词及增强词；bge_query 仍使用固定前缀 Represent this question for searching relevant passages: 。文档 embedding 不加该前缀。

## 7. Keyword 与 metadata filter

keyword_query 保持 required_terms、optional_terms 和 drug/disease/gene_protein/method/outcome buckets。MeSH record type 不能可靠映射细粒度实体类时保留在实体字段中，同时作为 disease bucket 的保守检索核心词。年份精确过滤仍为字符串 pub_year；范围、章节和不确定 article_type 继续进入 filter_plan。

## 8. 测试覆盖情况

共 {len(rows)} 条，MeSH 命中 query {mesh_cases} 条；ok={counts["ok"]}，warning={counts["warning"]}，invalid={counts["invalid"]}。覆盖 MI/myocardial infarction/heart attack、diabetes、breast/lung cancer、SARS、PCR、中文 fallback、年份、期刊、空和超长 query。

| raw_query | entities | sources | status |
| --- | ---: | --- | --- |
"""
    for row in rows:
        preview = row["raw_query"].replace("|", " ").replace("\n", " ")[:72] or "(empty)"
        text += f"| {preview} | {row['entity_count']} | {row['terminology_sources']} | {row['status']} |\n"
    text += """

## 9. 当前局限与下一步

MeSH 主要为英文主题词，中文映射仍是项目 seed；n-gram 字典匹配没有上下文消歧和拼写纠正；MeSH record type 不等于完整临床 NER 类型。下一步接入 Chroma 增强检索，并逐步增加 UMLS、hybrid retrieval、reranker 与 query rewrite。
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline MeSH query-understanding regression examples.")
    parser.add_argument("--terminology", default="artifacts/terminology/mesh_2026/medical_synonyms_mesh.jsonl")
    parser.add_argument("--term_index", default="artifacts/terminology/mesh_2026/term_to_concept_mesh.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    terminology, index = (root / args.terminology).resolve(), (root / args.term_index).resolve()
    resources = load_mesh_resources(terminology, index)
    if resources is None:
        print(f"[ERROR] MeSH terminology/index missing: {terminology} | {index}")
        return 2
    tables, reports, logs = root / "artifacts/metrics/t010_mesh_query_understanding", root / "reports/formal", root / "logs"
    for directory in (tables, reports, logs): directory.mkdir(parents=True, exist_ok=True)
    tee, old = Tee(logs / "17_test_mesh_query_understanding_mesh.log"), sys.stdout
    sys.stdout = tee
    try:
        rows = [make_row(query, resources) for query in TEST_QUERIES]
        csv_path, jsonl_path, report_path = tables / "query_understanding_examples_mesh.csv", tables / "query_understanding_examples_mesh.jsonl", reports / "MeSH查询理解与增强模块说明_mesh.md"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows: handle.write(dumps(row) + "\n")
        stats = json.loads((tables / "medical_terminology_stats_mesh.json").read_text(encoding="utf-8"))
        make_report(report_path, rows, terminology, index, stats)
        for number, row in enumerate(rows, 1):
            print(f"[CASE {number:02d}] status={row['status']} entities={row['entity_count']} sources={row['terminology_sources']}")
        print(f"[DONE] csv={csv_path}\n[DONE] jsonl={jsonl_path}\n[DONE] report={report_path}")
        return 0
    finally:
        sys.stdout = old
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
