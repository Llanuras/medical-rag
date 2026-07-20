from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from medical_rag.common.pmc import STOPWORDS, STRUCTURED_MARKERS, ensure_output_dirs, read_records_csv, setup_tee, write_markdown


def sample_group(df, label, n=5):
    group = df[df["length_group"] == label]
    if len(group) <= n:
        return group
    return group.sample(n=n, random_state=42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output_prefix", default="limit500")
    args = parser.parse_args()
    ensure_output_dirs()
    log_path = Path(f"logs/04_domain_sampling_review_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        records = read_records_csv(Path(args.records))
        tokens = read_records_csv(Path(args.tokens))
        df = records.merge(tokens[["record_id", "title_abstract_token_len"]], on="record_id", how="left")
        df["title_abstract_token_len"] = pd.to_numeric(df["title_abstract_token_len"], errors="coerce").fillna(0)
        q33 = df["title_abstract_token_len"].quantile(1/3)
        q66 = df["title_abstract_token_len"].quantile(2/3)
        def group(length):
            if length <= q33:
                return "short"
            if length <= q66:
                return "medium"
            return "long"
        df["length_group"] = df["title_abstract_token_len"].apply(group)
        samples = []
        for label in ["short", "medium", "long"]:
            s = sample_group(df, label, 5)
            samples.append(s)
            s.to_csv(f"reports/samples/sample_{label}_{args.output_prefix}.csv", index=False)
        sample_df = pd.concat(samples, ignore_index=True)
        md_lines = ["# 短中长文本分层抽样人工阅读样本", ""]
        for _, row in sample_df.iterrows():
            md_lines.extend([
                f"## {row['length_group']} | {row['record_id']}",
                f"- title: {row['title']}",
                f"- journal: {row['journal']}",
                f"- pub_year: {row['pub_year']}",
                f"- pmid: {row['pmid']}",
                f"- pmcid: {row['pmcid']}",
                f"- token_length: {int(row['title_abstract_token_len'])}",
                "",
                str(row["abstract"]),
                "",
            ])
        sample_md = Path(f"reports/samples/sample_short_medium_long_for_review_{args.output_prefix}.md")
        sample_md.write_text("\n".join(md_lines), encoding="utf-8")
        marker_rows = []
        abstracts_upper = records["abstract"].astype(str).str.upper()
        for marker in STRUCTURED_MARKERS:
            count = int(abstracts_upper.str.contains(rf"\b{marker}\b", regex=True).sum())
            marker_rows.append({"marker": marker, "count": count, "rate": count / max(1, len(records))})
        marker_path = Path(f"artifacts/metrics/t002_corpus_analysis/structured_abstract_markers_{args.output_prefix}.csv")
        pd.DataFrame(marker_rows).to_csv(marker_path, index=False)
        text = "\n".join(records["abstract"].astype(str).tolist())
        abbr = Counter(re.findall(r"\b[A-Z]{2,10}\b", text))
        abbr_path = Path(f"artifacts/metrics/t002_corpus_analysis/abbreviation_top50_{args.output_prefix}.csv")
        pd.DataFrame([{"abbreviation": k, "count": v} for k, v in abbr.most_common(50)]).to_csv(abbr_path, index=False)
        words = [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", text)]
        freq = Counter([w for w in words if w not in STOPWORDS])
        terms_path = Path(f"artifacts/metrics/t002_corpus_analysis/high_freq_terms_{args.output_prefix}.csv")
        pd.DataFrame([{"term": k, "count": v} for k, v in freq.most_common(50)]).to_csv(terms_path, index=False)
        top_abbr = ", ".join([k for k, _ in abbr.most_common(10)]) or "当前样本未发现明显大写缩写"
        top_terms = ", ".join([k for k, _ in freq.most_common(10)]) or "当前样本未统计到高频词"
        total = len(records)
        body = f"""
## 1. 本阶段分析目标

RAG 前需要理解医学文本的语言风格、信息密度和结构特点，以便设计合适的 prompt、query 改写、metadata 保留和 citation 策略。

## 2. 分层抽样策略

本阶段按 `title_abstract_token_len` 将 `{total}` 篇文献分为短、中、长三组，每组抽 5 篇，共 15 篇。短文本可能信息不足，中等文本通常适合作为摘要级检索单元，长文本可能需要切分或人工确认结构。

人工阅读样本保存在：`reports/samples/sample_short_medium_long_for_review_{args.output_prefix}.md`。

## 3. 医学文本结构分析

结构化摘要标记统计保存在 `artifacts/metrics/t002_corpus_analysis/structured_abstract_markers_{args.output_prefix}.csv`。BACKGROUND/METHODS/RESULTS/CONCLUSIONS 等标记可帮助后续按语义结构切分，尤其适合结构化摘要或全文章节处理。

## 4. 术语和缩写分析

医学文献常包含疾病名、药物名、基因名、统计术语和大写缩写。当前样本中 top 缩写包括：`{top_abbr}`。缩写会影响检索召回，例如同一概念可能存在全称、缩写和同义表达，因此后续 query 改写和 prompt 中应保留术语上下文。

## 5. 高频词分析

当前 abstract 高频词 top 项包括：`{top_terms}`。高频词反映当前样本主题分布，但普通科研词汇不应直接被误判为医学核心术语。

## 6. 人工阅读建议

请人工打开 `reports/samples/sample_short_medium_long_for_review_{args.output_prefix}.md`，重点观察短、中、长样本的信息密度、摘要结构、缩写密度和是否适合整体入库。

## 7. 对 RAG 的影响

领域语言特点会影响 prompt 设计、query 改写、embedding 模型选择、metadata 保留和 answer citation。后续回答应尽量保留 PMID/PMCID/source_file，以支持追溯。
"""
        doc_path = Path(f"reports/technical/04_domain_language_strategy_notes_{args.output_prefix}.md")
        write_markdown(doc_path, "医学文本领域内容理解说明", body)
        print("Wrote samples", sample_md)
        print("Wrote", marker_path)
        print("Wrote", abbr_path)
        print("Wrote", terms_path)
        print("Wrote", doc_path)

if __name__ == "__main__":
    main()
