from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

from medical_rag.common.pmc import ensure_output_dirs, read_records_csv, setup_tee, token_chunk_estimate, write_markdown

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def stats_for(series: pd.Series, label: str) -> dict:
    s = series.astype(int)
    return {
        "text_field": label,
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": int(s.min()),
        "max": int(s.max()),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "over_512_count": int((s > 512).sum()),
        "over_512_rate": float((s > 512).mean()),
        "over_1024_count": int((s > 1024).sum()),
        "over_1024_rate": float((s > 1024).mean()),
    }


def save_hist(series, path, title):
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=40, color="#4C78A8", edgecolor="white")
    plt.title(title)
    plt.xlabel("Token length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_box(series, path, title):
    plt.figure(figsize=(8, 2.8))
    plt.boxplot(series, vert=False)
    plt.title(title)
    plt.xlabel("Token length")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_prefix", default="limit500")
    args = parser.parse_args()
    ensure_output_dirs()
    log_path = Path(f"logs/03_token_length_analysis_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        df = read_records_csv(Path(args.input))
        print("Loading tokenizer", MODEL)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        def count_tokens(text):
            if not isinstance(text, str) or not text.strip():
                return 0
            return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        df["title_abstract_token_len"] = df["text_title_abstract"].apply(count_tokens)
        df["full_token_len"] = df["text_full"].apply(count_tokens)
        df["exceeds_512_title_abstract"] = df["title_abstract_token_len"] > 512
        df["exceeds_512_full"] = df["full_token_len"] > 512
        df["estimated_chunks_title_abstract"] = df["title_abstract_token_len"].apply(token_chunk_estimate)
        df["estimated_chunks_full"] = df["full_token_len"].apply(token_chunk_estimate)
        keep_cols = ["record_id", "source_file", "title_abstract_token_len", "full_token_len", "exceeds_512_title_abstract", "exceeds_512_full", "estimated_chunks_title_abstract", "estimated_chunks_full"]
        records_path = Path(f"artifacts/metrics/t002_corpus_analysis/token_length_records_{args.output_prefix}.csv")
        df[keep_cols].to_csv(records_path, index=False)
        stats = pd.DataFrame([
            stats_for(df["title_abstract_token_len"], "text_title_abstract"),
            stats_for(df["full_token_len"], "text_full"),
        ])
        stats_path = Path(f"artifacts/metrics/t002_corpus_analysis/token_length_stats_{args.output_prefix}.csv")
        stats.to_csv(stats_path, index=False)
        save_hist(df["title_abstract_token_len"], f"reports/figures/title_abstract_token_length_hist_{args.output_prefix}.png", "Title + abstract token length")
        save_box(df["title_abstract_token_len"], f"reports/figures/title_abstract_token_length_box_{args.output_prefix}.png", "Title + abstract token length")
        save_hist(df["full_token_len"], f"reports/figures/full_text_token_length_hist_{args.output_prefix}.png", "Full text token length")
        total = len(df)
        ta = stats[stats.text_field == "text_title_abstract"].iloc[0]
        full = stats[stats.text_field == "text_full"].iloc[0]
        if ta["p95"] <= 512 and ta["over_512_rate"] <= 0.05:
            recommendation = "title+abstract 大部分不超过 512 tokens，建议整体不切分，仅对超过 512 的长尾样本做 RecursiveCharacterTextSplitter。"
        else:
            recommendation = "title+abstract 存在明显超长比例，建议使用长尾滑动窗口或统一切分。"
        body = f"""
## 1. 本阶段分析目标

Token 长度决定文本是否能直接进入 embedding 模型。RAG 构建前需要统计长度分布，避免超出模型输入限制导致截断或信息丢失。

## 2. 为什么使用 embedding 模型对应 tokenizer

字符数、词数和 token 数并不等价。Embedding 模型通常按 tokenizer 的 token 序列处理输入，因此本阶段使用 `{MODEL}` 对应 tokenizer，并按 512 tokens 作为关键参考上限。

## 3. 分析对象选择

本阶段同时分析 `title+abstract` 和 `title+abstract+body`。初始 RAG 优先考虑 `title+abstract`，因为摘要信息密度高、长度可控；全文适合后续更复杂的章节切分和检索。

## 4. 长度分布指标解释

mean 和 median 描述整体长度水平；p95 和 p99 用于判断绝大多数样本是否能直接入库；max 反映长尾风险。如果 p95 接近或超过 512，则需要更积极的切分策略。

## 5. 可选策略

- 整体不切分：优点是上下文完整、实现简单；缺点是超长文本可能超过 embedding 限制。
- 只对长尾样本切分：优点是保留大多数摘要完整性；缺点是实现略复杂。
- 全量统一滑动窗口切分：优点是格式统一；缺点是短摘要会被不必要切分。
- 全文先章节切分再窗口切分：优点是适合结构清晰论文；缺点是依赖章节结构，XML 不统一时实现复杂。

## 6. 当前 `{total}` 篇真实数据结果

- title+abstract p95：`{ta['p95']:.2f}`
- title+abstract p99：`{ta['p99']:.2f}`
- title+abstract 超过 512 tokens 比例：`{ta['over_512_rate']:.4%}`
- full text p95：`{full['p95']:.2f}`
- full text p99：`{full['p99']:.2f}`
- full text 超过 512 tokens 比例：`{full['over_512_rate']:.4%}`

## 7. 初步策略选择

{recommendation}

对于全文 RAG，当前结果显示全文通常远长于摘要，因此正式使用 body 时应采用章节切分 + recursive split，而不是整体 embedding。
"""
        doc_path = Path(f"reports/technical/03_token_length_strategy_notes_{args.output_prefix}.md")
        write_markdown(doc_path, "Token 长度分析与 embedding 输入限制说明", body)
        print("Wrote", records_path)
        print("Wrote", stats_path)
        print("Wrote figures")
        print("Wrote", doc_path)

if __name__ == "__main__":
    main()
