from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset

from medical_rag.common.pmc import FIELD_COLUMNS, ensure_output_dirs, parse_pmc_xml, setup_tee, write_jsonl, write_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--data_dir", default="data/raw/pmc_oa_comm")
    parser.add_argument("--output_prefix", default="limit500")
    args = parser.parse_args()
    ensure_output_dirs()
    log_path = Path(f"logs/01_parse_pmc_xml_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        data_dir = Path(args.data_dir)
        xml_files = sorted(data_dir.rglob("*.xml"))[: args.limit]
        records = []
        failures = []
        print(f"Data dir: {data_dir}")
        print(f"Target limit: {args.limit}")
        print(f"XML selected: {len(xml_files)}")
        for idx, path in enumerate(xml_files, 1):
            record_id = f"pmc_{idx:06d}"
            try:
                records.append(parse_pmc_xml(path, record_id, source_root=data_dir))
            except Exception as exc:
                failures.append({"record_id": record_id, "source_file": str(path), "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {path}: {exc}")
        ds = Dataset.from_list(records)
        print(ds)
        df = pd.DataFrame(records, columns=FIELD_COLUMNS)
        csv_path = Path(f"artifacts/datasets/records/pmc_records_{args.output_prefix}.csv")
        jsonl_path = Path(f"artifacts/datasets/records/pmc_records_{args.output_prefix}.jsonl")
        df.to_csv(csv_path, index=False)
        write_jsonl(records, jsonl_path)
        summary = pd.DataFrame([
            {"metric": "data_dir", "value": str(data_dir)},
            {"metric": "target_limit", "value": args.limit},
            {"metric": "selected_xml", "value": len(xml_files)},
            {"metric": "parsed_success", "value": len(records)},
            {"metric": "parsed_failed", "value": len(failures)},
            {"metric": "dataset_num_rows", "value": ds.num_rows},
            {"metric": "dataset_columns", "value": ",".join(ds.column_names)},
        ])
        summary_path = Path(f"artifacts/metrics/t002_corpus_analysis/parse_summary_{args.output_prefix}.csv")
        summary.to_csv(summary_path, index=False)
        if failures:
            pd.DataFrame(failures).to_csv(f"artifacts/metrics/t002_corpus_analysis/parse_failures_{args.output_prefix}.csv", index=False)
        body = f"""
## 本阶段分析目标

本阶段将本地 PMC OA `oa_comm/xml` 文件解析为结构化 records，作为后续字段质量、token 长度、领域语言和 Chroma 入库分析的统一数据基础。

## 解析方式

- 数据来源：`{data_dir}`
- 目标处理数量：`{args.limit}`
- 实际选择 XML：`{len(xml_files)}`
- 解析成功：`{len(records)}`
- 解析失败：`{len(failures)}`

脚本使用本地 XML parser 提取字段，并通过 `datasets.Dataset.from_list(records)` 构建本地 HuggingFace Dataset 对象，体现 data pipeline。该流程不会从 HuggingFace 下载其他医学数据集。

## 与直接 load_dataset 在线数据集相比

使用本地 XML 的优点是严格符合 PMC OA `oa_comm` 数据源要求，可保留 source_file、pmid、pmcid 等溯源字段，并能观察真实 XML 结构差异。缺点是需要自行处理字段缺失、标签嵌套、正文过长和结构不统一问题。

直接使用在线 `load_dataset` 的优点是更方便、结构更统一；缺点是可能不符合本任务指定数据源，且难以确认样本确实来自当前本地 PMC OA `oa_comm` XML。

## 核心字段选择原因

- `title`：可增强检索文本，帮助短摘要样本提供主题信息。
- `abstract`：初始 RAG 最核心文本，信息密度高且长度通常可控。
- `body`：后续全文 RAG 的候选文本，但长度较长，需要切分策略。
- `journal`：未来可作为期刊 metadata filter。
- `pub_date/pub_year`：未来可作为时间过滤器，例如近 5 年文献检索。
- `pmid/pmcid`：用于 PubMed/PMC 原文追溯和回答 citation/source tracking。
- `source_file`：用于调试解析问题和本地数据溯源。

## XML 解析风险

PMC XML 可能存在字段缺失、结构标签不统一、结构化摘要嵌套复杂、正文过长、部分文献缺少 PMID 或期刊信息等问题。因此后续所有分析必须基于真实解析结果，不能写死或编造结论。
"""
        write_markdown(Path(f"reports/technical/01_parse_strategy_notes_{args.output_prefix}.md"), "PMC XML 解析策略说明", body)
        print(f"Wrote {csv_path}")
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {summary_path}")

if __name__ == "__main__":
    main()
