from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from medical_rag.common.pmc import ensure_output_dirs, read_records_csv, setup_tee, token_chunk_estimate, write_markdown

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
SECTION_PATTERNS = {
    "introduction": re.compile(r"\b(introduction|background)\b", re.I),
    "methods": re.compile(r"\b(methods?|materials and methods|methodology)\b", re.I),
    "results": re.compile(r"\b(results?)\b", re.I),
    "discussion": re.compile(r"\b(discussion)\b", re.I),
    "conclusion": re.compile(r"\b(conclusions?|summary)\b", re.I),
}


def xml_section_titles(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml-xml")
    body = soup.find("body")
    if body is None:
        return []
    titles = []
    for sec in body.find_all("sec"):
        title = sec.find("title", recursive=False)
        if title:
            text = " ".join(title.get_text(" ", strip=True).split())
            if text:
                titles.append(text)
    return titles


def marker_flags(titles: list[str]) -> dict[str, bool]:
    joined = "\n".join(titles)
    return {name: bool(pattern.search(joined)) for name, pattern in SECTION_PATTERNS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--data_dir", default="data/raw/pmc_oa_comm")
    parser.add_argument("--output_prefix", default="limit3028")
    args = parser.parse_args()

    ensure_output_dirs()
    log_path = Path(f"logs/06_full_text_split_strategy_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        records = read_records_csv(Path(args.records))
        tokens = pd.read_csv(args.tokens).fillna("")
        df = records.merge(
            tokens[["record_id", "full_token_len", "estimated_chunks_full"]],
            on="record_id",
            how="left",
        )
        df["full_token_len"] = pd.to_numeric(df["full_token_len"], errors="coerce").fillna(0).astype(int)
        df["estimated_chunks_full"] = pd.to_numeric(df["estimated_chunks_full"], errors="coerce").fillna(0).astype(int)

        rows = []
        title_counter: Counter[str] = Counter()
        data_dir = Path(args.data_dir)
        for _, row in df.iterrows():
            source_file = str(row["source_file"])
            xml_path = data_dir / source_file
            titles: list[str] = []
            if xml_path.exists():
                titles = xml_section_titles(xml_path)
            title_counter.update(t.lower() for t in titles)
            flags = marker_flags(titles)
            rows.append(
                {
                    "record_id": row["record_id"],
                    "source_file": source_file,
                    "full_token_len": row["full_token_len"],
                    "estimated_chunks_full": row["estimated_chunks_full"],
                    "section_title_count": len(titles),
                    "has_any_section_title": bool(titles),
                    **{f"has_{k}": v for k, v in flags.items()},
                }
            )

        section_df = pd.DataFrame(rows)
        section_path = Path(f"artifacts/metrics/t002_corpus_analysis/full_text_section_analysis_{args.output_prefix}.csv")
        section_df.to_csv(section_path, index=False)

        title_path = Path(f"artifacts/metrics/t002_corpus_analysis/full_text_section_title_top50_{args.output_prefix}.csv")
        pd.DataFrame(
            [{"section_title": title, "count": count} for title, count in title_counter.most_common(50)]
        ).to_csv(title_path, index=False)

        total = len(df)
        full_tokens = df["full_token_len"]
        chunk_counts = df["estimated_chunks_full"]
        summary_rows = [
            {"metric": "records", "value": total},
            {"metric": "chunk_size", "value": CHUNK_SIZE},
            {"metric": "chunk_overlap", "value": CHUNK_OVERLAP},
            {"metric": "full_token_mean", "value": float(full_tokens.mean())},
            {"metric": "full_token_median", "value": float(full_tokens.median())},
            {"metric": "full_token_p95", "value": float(full_tokens.quantile(0.95))},
            {"metric": "full_token_p99", "value": float(full_tokens.quantile(0.99))},
            {"metric": "full_over_512_count", "value": int((full_tokens > 512).sum())},
            {"metric": "full_over_512_rate", "value": float((full_tokens > 512).mean())},
            {"metric": "estimated_total_full_chunks", "value": int(chunk_counts.sum())},
            {"metric": "estimated_chunks_mean", "value": float(chunk_counts.mean())},
            {"metric": "estimated_chunks_p95", "value": float(chunk_counts.quantile(0.95))},
            {"metric": "records_with_section_titles", "value": int(section_df["has_any_section_title"].sum())},
            {"metric": "records_with_section_titles_rate", "value": float(section_df["has_any_section_title"].mean())},
            {"metric": "records_with_introduction_marker", "value": int(section_df["has_introduction"].sum())},
            {"metric": "records_with_methods_marker", "value": int(section_df["has_methods"].sum())},
            {"metric": "records_with_results_marker", "value": int(section_df["has_results"].sum())},
            {"metric": "records_with_discussion_marker", "value": int(section_df["has_discussion"].sum())},
            {"metric": "records_with_conclusion_marker", "value": int(section_df["has_conclusion"].sum())},
        ]
        summary = pd.DataFrame(summary_rows)
        summary_path = Path(f"artifacts/metrics/t002_corpus_analysis/full_text_split_strategy_summary_{args.output_prefix}.csv")
        summary.to_csv(summary_path, index=False)

        section_rate = float(section_df["has_any_section_title"].mean())
        over512 = float((full_tokens > 512).mean())
        total_chunks = int(chunk_counts.sum())
        p95 = float(full_tokens.quantile(0.95))
        p99 = float(full_tokens.quantile(0.99))
        body = f"""
## 1. 本阶段分析目标

本阶段专门针对全文 `title + abstract + body` 制定切分策略。此前摘要级 Chroma 测试只使用 `title + abstract`，但全文 RAG 必须处理正文长度、章节结构和 chunk 数量增长问题。

## 2. 当前 `{total}` 篇全文长度结论

- full text p95：`{p95:.2f}` tokens
- full text p99：`{p99:.2f}` tokens
- full text 超过 512 tokens 比例：`{over512:.4%}`
- 按 `chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}` 估算全文总 chunks：`{total_chunks}`

这些结果说明全文不能整体送入 embedding 模型，必须切分。

## 3. XML 正文章节结构

- 检测到正文 section title 的文献比例：`{section_rate:.4%}`
- 章节标题统计见 `artifacts/metrics/t002_corpus_analysis/full_text_section_title_top50_{args.output_prefix}.csv`
- 每篇文章的章节标记见 `artifacts/metrics/t002_corpus_analysis/full_text_section_analysis_{args.output_prefix}.csv`

如果正文有明确章节标题，优先保留章节 metadata；如果章节标题缺失或结构不统一，再退回到递归滑动窗口。

## 4. 可选全文切分策略

- 全文整体不切分：不适合当前数据，因为绝大多数全文超过 512 tokens。
- 全量统一滑动窗口：实现简单，能稳定控制输入长度；缺点是可能切断章节语义。
- 章节优先 + 递归切分：先按 XML `sec/title` 保留章节，再对超长章节使用 recursive split；优点是更符合医学论文结构，缺点是实现更复杂。
- 仅摘要入库，正文作为补充：实现成本最低，但不能回答正文细节问题。

## 5. 推荐策略

当前推荐：正式全文 RAG 使用“章节优先 + RecursiveCharacterTextSplitter”的混合策略。

- 文本来源：`title + abstract + section_title + section_text`
- chunk 参数：`chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}`
- metadata：`record_id, source_file, title, journal, pub_year, pmid, pmcid, article_type, section_title, chunk_index, chunk_count`
- 对没有章节标题的正文：直接对全文 body 做 recursive split
- 对摘要级检索：继续保留 `title + abstract` 独立入库或作为高优先级字段

## 6. 对后续 RAG 的影响

全文入库会显著增加 chunk 数和 embedding 成本，但能覆盖摘要没有写出的实验细节、方法、结果和讨论。建议先做 50/100 篇全文 Chroma 小测试，再扩展到 `{total}` 篇全文。
"""
        doc_path = Path(f"reports/technical/06_full_text_split_strategy_notes_{args.output_prefix}.md")
        write_markdown(doc_path, "全文切分策略分析说明", body)

        print("Wrote", section_path)
        print("Wrote", title_path)
        print("Wrote", summary_path)
        print("Wrote", doc_path)


if __name__ == "__main__":
    main()
