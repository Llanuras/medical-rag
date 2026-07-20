from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from medical_rag.query.understanding import process_medical_query

TEST_QUERIES = [
    "二甲双胍对心血管疾病有何影响？", "MI treatment", "heart attack aspirin mortality",
    "EGFR mutation lung cancer treatment", "HIV reverse transcriptase inhibitor resistance",
    "type 2 diabetes insulin sensitivity after 2010", "PLoS ONE breast cancer gene expression",
    "SARS coronavirus spike protein", "PCR DNA amplification", "warfarin bleeding risk",
    "2010年关于肺癌EGFR突变的研究", "metformin cardiovascular outcome in PLoS ONE", "",
    "EGFR mutation " * 200,
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


def row_for(raw_query: str) -> dict[str, str]:
    result = process_medical_query(raw_query)
    status = "invalid" if not result.clean_query else ("warning" if result.warnings else "ok")
    return {
        "raw_query": raw_query, "clean_query": result.clean_query, "entity_count": str(len(result.entities)),
        "entities_json": dumps([asdict(entity) for entity in result.entities]), "expanded_terms": dumps(result.expanded_terms),
        "vector_query": result.vector_query, "bge_query": result.bge_query, "keyword_query_json": dumps(result.keyword_query),
        "where_filter_json": dumps(result.where_filter), "filter_plan_json": dumps(result.filter_plan),
        "warnings": dumps(result.warnings), "status": status,
    }


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    counts = {status: sum(row["status"] == status for row in rows) for status in ("ok", "warning", "invalid")}
    detected = sum(int(row["entity_count"]) > 0 for row in rows)
    text = f"""# 查询理解与增强模块说明

## 1. 本周任务目标

该模块是智能检索的前置层，将用户自然语言医学问题转换为可解释的结构化信息，供后续 BGE 向量检索、BM25/hybrid retrieval、Chroma metadata filter 和 reranker 使用。本轮不重建 embedding 或 Chroma 索引，也不生成 RAG 答案。

## 2. 模块输入输出

- 输入：中英文自然语言 query。
- 主入口：src/medical_rag/query/understanding.py 的 process_medical_query(query)。
- 输出：clean query、带位置的实体、保守同义词扩展、vector/BGE query、keyword query、精确 where_filter、filter_plan 和 warnings。

## 3. 基础清洗策略

- None、空白输入返回友好 warning，不继续处理。
- 压缩空白、统一常见中英文标点、压缩重复标点，保留缩写、数字和年份。
- 超过 1000 字符时截断并记录 warning。

## 4. 医学实体识别规则

采用离线静态词表和正则。英文为大小写不敏感的严格单词边界匹配，中文短语直接匹配；长短语优先，重叠和重复标准化实体会去除。覆盖 drug、disease、gene/protein、method、outcome。

## 5. 同义词词典设计

MEDICAL_SYNONYMS 覆盖 MI/heart attack/myocardial infarction、二甲双胍/metformin、心血管疾病/CVD、T2DM、EGFR、PCR、HIV 和 SARS。每个实体最多扩展 4 个词、保留原词并防止递归扩展；中文实体补充英文检索词。

## 6. 向量 query 与 BGE query

- vector_query 由实体原文、标准化术语与同义词去重拼接。
- bge_query 固定为 Represent this question for searching relevant passages: 加 vector_query，与既有文档 embedding 不加 instruction 的 BGE 约定一致。

## 7. 关键词 query 结构

required_terms 放 drug、disease、gene/protein；optional_terms 放同义词扩展；entity_terms 分为 drug、disease、gene_protein、method、outcome，可直接供 BM25/hybrid retrieval 使用。

## 8. Metadata filter 提取规则

- 单年（2010、2010年、in 2010）输出精确 Chroma filter，例如 {{"pub_year": "2010"}}；pub_year 始终为字符串。
- PLoS ONE、Nature、BMC 形成 journal filter；research article 形成 {{"article_type": "research-article"}}。
- after/since/before/from-to 年份范围及章节条件放入 filter_plan，不假设当前 Chroma 对字符串范围或 section 标签完全兼容。

## 9. 测试 query 覆盖情况

共 {len(rows)} 条，{detected} 条识别到实体；ok={counts["ok"]}，warning={counts["warning"]}，invalid={counts["invalid"]}。覆盖中文药物/疾病、MI、EGFR、HIV、SARS、PCR、warfarin、年份、期刊、空 query 与超长 query。

| raw_query | entity_count | status |
| --- | ---: | --- |
"""
    for row in rows:
        preview = row["raw_query"].replace("|", " ").replace("\n", " ")[:80] or "(empty)"
        text += f"| {preview} | {row['entity_count']} | {row['status']} |\n"
    text += """

## 10. 当前局限

- 词表不是完整 UMLS/MeSH，未做上下文消歧、拼写纠正或学习式 NER。
- 词表外专业术语会遗漏；宽泛 outcome 词仅作为轻量信号。
- 接 Chroma 前仍应用现有 metadata 样本复核 journal、article_type、section 的真实取值。

## 11. 下一步

接入 Chroma 增强检索，随后实现 hybrid retrieval、reranker 和 query rewrite；继续保持查询理解、检索、重排和答案生成模块分离。
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tables, reports, logs = root / "artifacts/metrics/t009_query_understanding", root / "reports/formal", root / "logs"
    for directory in (tables, reports, logs):
        directory.mkdir(parents=True, exist_ok=True)
    tee, old = Tee(logs / "15_test_query_understanding.log"), sys.stdout
    sys.stdout = tee
    try:
        rows = [row_for(query) for query in TEST_QUERIES]
        fields = list(rows[0])
        csv_path, jsonl_path, report_path = tables / "query_understanding_examples.csv", tables / "query_understanding_examples.jsonl", reports / "查询理解与增强模块说明.md"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_report(report_path, rows)
        for index, row in enumerate(rows, 1):
            print(f"[CASE {index:02d}] status={row['status']} entities={row['entity_count']} clean_query={row['clean_query']}")
        print(f"[DONE] csv={csv_path}\n[DONE] jsonl={jsonl_path}\n[DONE] report={report_path}")
        return 0
    finally:
        sys.stdout = old
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
