from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from medical_rag.common.pmc import TARGET_FIELDS, ensure_output_dirs, is_empty_value, non_empty, read_records_csv, safe_rate, setup_tee, simple_word_count, write_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_prefix", default="limit500")
    args = parser.parse_args()
    ensure_output_dirs()
    log_path = Path(f"logs/02_analyze_fields_quality_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        df = read_records_csv(Path(args.input))
        total = len(df)
        rows = []
        for field in TARGET_FIELDS:
            present = non_empty(df[field]) if field in df else pd.Series([False] * total)
            non_empty_count = int(present.sum())
            rows.append({"field": field, "total_count": total, "non_empty_count": non_empty_count, "missing_count": total - non_empty_count, "missing_rate": safe_rate(total - non_empty_count, total)})
        missing_df = pd.DataFrame(rows)
        missing_path = Path(f"artifacts/metrics/t002_corpus_analysis/missing_rate_{args.output_prefix}.csv")
        missing_df.to_csv(missing_path, index=False)
        abstract_missing_rate = float(missing_df.loc[missing_df.field == "abstract", "missing_rate"].iloc[0])
        flags = pd.DataFrame({"record_id": df["record_id"]})
        flags["too_short_abstract"] = df["abstract"].fillna("").apply(lambda x: len(str(x).strip()) < 50 or simple_word_count(str(x)) < 20)
        flags["empty_title"] = ~non_empty(df["title"])
        flags["empty_abstract"] = ~non_empty(df["abstract"])
        flags["empty_body"] = ~non_empty(df["body"])
        flags["encoding_issue"] = df[["title", "abstract", "body"]].astype(str).agg(" ".join, axis=1).str.contains("�", regex=False)
        combo = (df["title"].astype(str).str.strip() + "\n" + df["abstract"].astype(str).str.strip()).str.lower()
        flags["duplicate_title_abstract"] = combo.duplicated(keep=False) & combo.str.strip().ne("")
        def decision(row):
            if row.empty_title and row.empty_abstract and row.empty_body:
                return "drop_no_text"
            if row.encoding_issue or row.duplicate_title_abstract or (row.empty_abstract and row.empty_body):
                return "need_review"
            if row.too_short_abstract or row.empty_abstract or row.empty_body or row.empty_title:
                return "keep_with_warning"
            return "keep"
        flags["quality_decision"] = flags.apply(decision, axis=1)
        flags_path = Path(f"artifacts/metrics/t002_corpus_analysis/quality_flags_{args.output_prefix}.csv")
        flags.to_csv(flags_path, index=False)
        dup_pmid = int(df.loc[non_empty(df["pmid"]), "pmid"].duplicated(keep=False).sum()) if "pmid" in df else 0
        dup_pmcid = int(df.loc[non_empty(df["pmcid"]), "pmcid"].duplicated(keep=False).sum()) if "pmcid" in df else 0
        quality_rows = [
            {"metric": "total_records", "value": total},
            {"metric": "too_short_abstract_count", "value": int(flags["too_short_abstract"].sum())},
            {"metric": "empty_title_count", "value": int(flags["empty_title"].sum())},
            {"metric": "empty_abstract_count", "value": int(flags["empty_abstract"].sum())},
            {"metric": "empty_body_count", "value": int(flags["empty_body"].sum())},
            {"metric": "encoding_issue_count", "value": int(flags["encoding_issue"].sum())},
            {"metric": "duplicate_title_abstract_count", "value": int(flags["duplicate_title_abstract"].sum())},
            {"metric": "duplicate_pmid_count", "value": dup_pmid},
            {"metric": "duplicate_pmcid_count", "value": dup_pmcid},
        ]
        for value, count in flags["quality_decision"].value_counts().to_dict().items():
            quality_rows.append({"metric": f"quality_decision_{value}", "value": count})
        quality_path = Path(f"artifacts/metrics/t002_corpus_analysis/quality_summary_{args.output_prefix}.csv")
        pd.DataFrame(quality_rows).to_csv(quality_path, index=False)
        meta_rows = []
        usage = {
            "title": "增强检索文本，作为标题上下文",
            "journal": "metadata filter，支持按期刊过滤",
            "pub_year": "metadata filter，支持近年文献过滤",
            "pmid": "PubMed 原文追溯和 citation",
            "pmcid": "PMC 原文追溯和 citation",
            "source_file": "本地调试和数据溯源",
        }
        for field, future in usage.items():
            if field in df:
                availability = safe_rate(int(non_empty(df[field]).sum()), total)
            else:
                availability = 0.0
            risk = "可用率高，适合作为 metadata" if availability >= 0.9 else "可用率不足，使用时需要 fallback 或人工确认"
            recommendation = "保留" if availability > 0 else "当前样本未提取到，暂不依赖"
            meta_rows.append({"field": field, "availability_rate": availability, "future_rag_usage": future, "risk": risk, "recommendation": recommendation})
        meta_path = Path(f"artifacts/metrics/t002_corpus_analysis/metadata_summary_{args.output_prefix}.csv")
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
        get_missing = lambda f: float(missing_df.loc[missing_df.field == f, "missing_rate"].iloc[0]) if f in set(missing_df.field) else 1.0
        abstract_strategy = "直接丢弃缺失 abstract 的样本" if abstract_missing_rate <= 0.01 else "分层处理：body 替代、title-only metadata、无文本丢弃"
        body = f"""
## 1. 本阶段分析目标

RAG 构建前需要确认字段完整性、文本质量和 metadata 可用性。字段缺失会影响向量库文本构造，metadata 缺失会影响后续过滤和溯源，低质量文本会降低检索和回答质量。

## 2. 分析维度

本阶段分析了字段缺失率、abstract 可用性、基础文本质量、metadata 可用性和原文溯源能力。

## 3. 字段缺失的可选处理策略

- 直接丢弃缺失样本：优点是保证入库文本质量；缺点是可能损失样本量。
- 使用 body 替代 abstract：优点是保留文献信息；缺点是 body 较长且可能需要额外切分。
- 使用 title 作为弱文本：优点是保留主题信息；缺点是信息量不足，不适合作为核心向量文本。
- 保留为 metadata 但不入库：优点是保留追溯信息；缺点是不能参与语义检索。
- 人工补全或二次抓取：优点是质量高；缺点是成本较高，不适合当前批量准备阶段。

## 4. abstract 缺失处理策略

当前 `{total}` 篇中 abstract 缺失率为 `{abstract_missing_rate:.4%}`。因此建议：{abstract_strategy}。

## 5. 基础质量清洗策略

- 极短 abstract：标记为 `keep_with_warning`，后续入库时可保留但需人工抽查。
- 乱码文本：标记为 `need_review`，避免污染向量库。
- 重复 title+abstract：标记为 `need_review`，避免重复文档影响检索排序。
- 空正文：如果 title+abstract 可用，可用于摘要级 RAG；若核心文本为空则不入库。

## 6. 关键 metadata 字段策略

- title 缺失率：`{get_missing('title'):.4%}`，可用于增强检索文本。
- journal 可用率：`{1 - get_missing('journal'):.4%}`，可作为期刊过滤器；如果可用率不足，未来实现“检索近5年某一期刊上的文献”需要 fallback。
- pub_year 可用率：`{1 - get_missing('pub_year'):.4%}`，可作为时间过滤器。
- pmid 可用率：`{1 - get_missing('pmid'):.4%}`，可用于 PubMed 追溯。
- pmcid 可用率：`{1 - get_missing('pmcid'):.4%}`，可用于 PMC 追溯。

## 7. 当前 `{total}` 篇数据的实际结论

详见 `artifacts/metrics/t002_corpus_analysis/missing_rate_{args.output_prefix}.csv`、`artifacts/metrics/t002_corpus_analysis/quality_summary_{args.output_prefix}.csv` 和 `artifacts/metrics/t002_corpus_analysis/metadata_summary_{args.output_prefix}.csv`。所有数值均来自真实解析的 `{total}` 篇 XML。

## 8. 对后续 RAG 构建的影响

建议入库样本使用 `quality_decision` 为 `keep` 或 `keep_with_warning` 且 `text_title_abstract` 非空的记录。`journal`、`pub_year`、`pmid`、`pmcid`、`source_file` 应作为 metadata 保存，用于后续过滤、citation 和 source tracking。
"""
        doc_path = Path(f"reports/technical/02_field_quality_strategy_notes_{args.output_prefix}.md")
        write_markdown(doc_path, "字段完整性、清洗策略与关键字段分析说明", body)
        print("Wrote", missing_path)
        print("Wrote", quality_path)
        print("Wrote", flags_path)
        print("Wrote", meta_path)
        print("Wrote", doc_path)

if __name__ == "__main__":
    main()
