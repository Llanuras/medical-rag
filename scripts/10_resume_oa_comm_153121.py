from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer


def progress_line(label: str, current: int, total: int, started: float) -> str:
    width = 28
    ratio = 0.0 if total <= 0 else min(1.0, current / total)
    filled = int(width * ratio)
    elapsed = max(0.001, time.perf_counter() - started)
    rate = current / elapsed
    remaining = max(0, total - current)
    eta_seconds = remaining / rate if rate > 0 else 0
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"{label} [{bar}] {current}/{total} "
        f"({ratio:.1%}) rate={rate:.1f}/s eta={eta_seconds/60:.1f}min"
    )


def load_analyzer():
    path = Path(__file__).with_name("10_analyze_oa_comm_153121.py")
    spec = importlib.util.spec_from_file_location("oa_comm_153121_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def parse_text_only(path: Path, mod) -> tuple[str, str, list[str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml-xml")
    title = mod.first_text(soup, "article-title")
    abstract = mod.first_text(soup, "abstract")
    body_node = soup.find("body")
    body = mod.text_or_empty(body_node)
    titles = mod.xml_section_titles(body_node)
    text_full = "\n\n".join(part for part in [title.strip(), abstract.strip(), body.strip()] if part)
    return abstract, text_full, titles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/pmc_oa_comm")
    parser.add_argument("--output_prefix", default="limit153121")
    parser.add_argument("--expected_total", type=int, default=153121)
    parser.add_argument("--sample_per_group", type=int, default=8)
    args = parser.parse_args()

    mod = load_analyzer()
    project_dir = Path.cwd()
    data_dir = Path(args.data_dir)
    mod.ensure_project_hf_cache(project_dir)
    mod.ensure_output_dirs()
    Path("reports/formal").mkdir(parents=True, exist_ok=True)
    Path("artifacts/metrics/t006_fullscale_analysis").mkdir(parents=True, exist_ok=True)

    xml_files = sorted(data_dir.rglob("*.xml"))
    if len(xml_files) != args.expected_total:
        raise RuntimeError(f"Expected {args.expected_total} XML files, found {len(xml_files)}")

    light_path = Path(f"artifacts/metrics/t006_fullscale_analysis/pmc_records_light_{args.output_prefix}.csv")
    if not light_path.exists():
        raise RuntimeError(f"Cannot resume; missing {light_path}")

    started = time.perf_counter()
    existing_df = pd.read_csv(light_path, dtype=str, keep_default_na=False).fillna("")
    existing_count = int(existing_df.shape[0])
    if existing_count >= len(xml_files):
        start_index = len(xml_files) + 1
    else:
        expected_next = str(xml_files[existing_count].relative_to(data_dir))
        last_existing = str(existing_df.iloc[-1]["source_file"]) if existing_count else ""
        print(f"Existing rows: {existing_count}; last={last_existing}; next={expected_next}")
        start_index = existing_count + 1

    fieldnames = list(existing_df.columns)
    log_path = Path(f"logs/10_analyze_oa_comm_{args.output_prefix}.log")
    resume_log_path = Path(f"logs/10_resume_oa_comm_{args.output_prefix}.log")

    tokenizer_ref = mod.resolve_local_tokenizer(project_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_ref), local_files_only=True)
    failures: list[dict[str, object]] = []

    with light_path.open("a", encoding="utf-8", newline="") as light_f, resume_log_path.open("a", encoding="utf-8") as resume_log:
        writer = csv.DictWriter(light_f, fieldnames=fieldnames)

        def log_print(message: str) -> None:
            print(message)
            resume_log.write(message + "\n")
            resume_log.flush()

        log_print(f"RESUME START {datetime.now().isoformat(timespec='seconds')}")
        log_print(f"Existing rows: {existing_count}")
        log_print(f"Resume from XML index: {start_index}")
        log_print(f"Tokenizer: {tokenizer_ref}")
        log_print(progress_line("Resume rows", existing_count, len(xml_files), started))

        for idx in range(start_index, len(xml_files) + 1):
            path = xml_files[idx - 1]
            record_id = f"pmc_{idx:06d}"
            try:
                row, _ = mod.parse_light_record(path, record_id, data_dir, tokenizer)
            except Exception as exc:
                failures.append(
                    {
                        "record_id": record_id,
                        "source_file": str(path.relative_to(data_dir)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if idx % 1000 == 0:
                    log_print(f"{progress_line('Resume rows', idx, len(xml_files), started)} failures={len(failures)}")
                continue
            writer.writerow({name: row.get(name, "") for name in fieldnames})
            if idx % 1000 == 0 or idx == len(xml_files):
                log_print(f"{progress_line('Resume rows', idx, len(xml_files), started)} failures={len(failures)}")

    rows_df = pd.read_csv(light_path, dtype=str, keep_default_na=False).fillna("")
    parsed_success = int(rows_df.shape[0])
    parsed_failed = len(failures)
    if parsed_success + parsed_failed != len(xml_files):
        raise RuntimeError(f"Count mismatch: success={parsed_success}, failed={parsed_failed}, xml={len(xml_files)}")

    bool_cols = [
        "title_missing",
        "abstract_missing",
        "body_missing",
        "exceeds_512_title_abstract",
        "exceeds_512_full",
        "has_any_section_title",
        "has_introduction",
        "has_methods",
        "has_results",
        "has_discussion",
        "has_conclusion",
        "imrad_core",
        "imrad_with_conclusion",
    ]
    for col in bool_cols:
        rows_df[col] = rows_df[col].astype(str).str.lower().eq("true")
    for col in ["title_abstract_token_len", "full_token_len", "estimated_chunks_title_abstract", "estimated_chunks_full", "section_title_count"]:
        rows_df[col] = pd.to_numeric(rows_df[col], errors="coerce").fillna(0).astype(int)

    lexical_start = time.perf_counter()
    section_title_counter = Counter()
    structured_marker_counter = Counter()
    unigram_counter: Counter[tuple[str, ...]] = Counter()
    abbr_counter: Counter[str] = Counter()
    for idx, path in enumerate(xml_files, start=1):
        abstract, text_full, titles = parse_text_only(path, mod)
        section_title_counter.update(t.lower() for t in titles)
        abstract_upper = abstract.upper()
        for marker in mod.STRUCTURED_MARKERS:
            if re.search(rf"\b{marker}\b", abstract_upper):
                structured_marker_counter[marker] += 1
        toks = mod.filtered_words(text_full)
        mod.update_ngram_counter(unigram_counter, toks, 1)
        mod.update_abbreviations(abbr_counter, text_full)
        if idx % 5000 == 0 or idx == len(xml_files):
            print(progress_line("Lexical pass", idx, len(xml_files), lexical_start), flush=True)

    metadata_nonempty = Counter()
    missing_rows = []
    field_map = {
        "title": "title",
        "journal": "journal",
        "pub_date": "pub_date",
        "pub_year": "pub_year",
        "pmid": "pmid",
        "pmcid": "pmcid",
        "article_type": "article_type",
    }
    for field in mod.TARGET_FIELDS:
        if field == "abstract":
            non_empty = int((~rows_df["abstract_missing"]).sum())
        elif field == "body":
            non_empty = int((~rows_df["body_missing"]).sum())
        else:
            col = field_map.get(field)
            non_empty = int(rows_df[col].astype(str).str.strip().ne("").sum()) if col else 0
        metadata_nonempty[field] = non_empty
        missing = parsed_success - non_empty
        missing_rows.append(
            {
                "field": field,
                "total_count": parsed_success,
                "non_empty_count": non_empty,
                "missing_count": missing,
                "missing_rate": mod.safe_rate(missing, parsed_success),
            }
        )

    pmid_counts = rows_df.loc[rows_df["pmid"].str.strip().ne(""), "pmid"].value_counts()
    pmcid_counts = rows_df.loc[rows_df["pmcid"].str.strip().ne(""), "pmcid"].value_counts()
    quality_counts = rows_df["quality_decision"].value_counts().to_dict()
    route_counts = rows_df["recommended_split_strategy"].value_counts().to_dict()
    route_est_chunks = rows_df.groupby("recommended_split_strategy")["estimated_chunks_full"].sum().to_dict()

    pd.DataFrame(
        [
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
    ).to_csv(f"artifacts/metrics/t006_fullscale_analysis/parse_summary_{args.output_prefix}.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(f"artifacts/metrics/t006_fullscale_analysis/missing_rate_{args.output_prefix}.csv", index=False)
    quality_rows = [
        {"metric": "total_records", "value": parsed_success},
        {"metric": "empty_title_count", "value": int(rows_df["title_missing"].sum())},
        {"metric": "empty_abstract_count", "value": int(rows_df["abstract_missing"].sum())},
        {"metric": "empty_body_count", "value": int(rows_df["body_missing"].sum())},
        {"metric": "duplicate_pmid_count", "value": int(pmid_counts[pmid_counts > 1].sum())},
        {"metric": "duplicate_pmcid_count", "value": int(pmcid_counts[pmcid_counts > 1].sum())},
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
        availability = 1.0 if field == "source_file" else mod.safe_rate(int(metadata_nonempty[field]), parsed_success)
        meta_rows.append({"field": field, "availability_rate": availability, "future_rag_usage": future, "recommendation": "保留" if availability > 0 else "暂不依赖"})
    pd.DataFrame(meta_rows).to_csv(f"artifacts/metrics/t006_fullscale_analysis/metadata_summary_{args.output_prefix}.csv", index=False)

    token_stats = [
        mod.stats_for(rows_df["title_abstract_token_len"].tolist(), "text_title_abstract"),
        mod.stats_for(rows_df["full_token_len"].tolist(), "text_full"),
    ]
    pd.DataFrame(token_stats).to_csv(f"artifacts/metrics/t006_fullscale_analysis/token_length_stats_{args.output_prefix}.csv", index=False)
    rows_df[[
        "record_id",
        "source_file",
        "title_abstract_token_len",
        "full_token_len",
        "exceeds_512_title_abstract",
        "exceeds_512_full",
        "estimated_chunks_title_abstract",
        "estimated_chunks_full",
        "recommended_split_strategy",
    ]].to_csv(f"artifacts/metrics/t006_fullscale_analysis/token_length_records_light_{args.output_prefix}.csv", index=False)
    rows_df[[
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
    ]].to_csv(f"artifacts/metrics/t006_fullscale_analysis/full_text_section_analysis_light_{args.output_prefix}.csv", index=False)
    pd.DataFrame([{"section_title": k, "count": v} for k, v in section_title_counter.most_common(80)]).to_csv(
        f"artifacts/metrics/t006_fullscale_analysis/full_text_section_title_top80_{args.output_prefix}.csv", index=False
    )
    split_rows = [
        {"metric": "records", "value": parsed_success},
        {"metric": "chunk_size_for_strategy", "value": mod.CHUNK_SIZE},
        {"metric": "chunk_overlap_for_strategy", "value": mod.CHUNK_OVERLAP},
        {"metric": "whole_doc_token_limit", "value": mod.WHOLE_DOC_TOKEN_LIMIT},
        {"metric": "full_token_mean", "value": float(rows_df["full_token_len"].mean())},
        {"metric": "full_token_median", "value": float(rows_df["full_token_len"].median())},
        {"metric": "full_token_p95", "value": float(rows_df["full_token_len"].quantile(0.95))},
        {"metric": "full_token_p99", "value": float(rows_df["full_token_len"].quantile(0.99))},
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

    pd.DataFrame(mod.top_ngrams(unigram_counter)).to_csv(f"artifacts/metrics/t006_fullscale_analysis/fulltext_high_freq_unigrams_{args.output_prefix}.csv", index=False)
    pd.DataFrame(
        [
            {
                "term": "SKIPPED_FULLSCALE_BIGRAM_LIGHT_LEXICAL_PASS",
                "count": "",
                "note": "全量 bigram 在轻量 lexical pass 中跳过，以避免 15w 全文 Counter 长尾导致运行过慢。",
            }
        ]
    ).to_csv(f"artifacts/metrics/t006_fullscale_analysis/fulltext_high_freq_bigrams_{args.output_prefix}.csv", index=False)
    pd.DataFrame(
        [
            {
                "term": "SKIPPED_FULLSCALE_TRIGRAM_LIGHT_LEXICAL_PASS",
                "count": "",
                "note": "全量 trigram 在轻量 lexical pass 中跳过，以避免 15w 全文 Counter 长尾导致运行过慢。",
            }
        ]
    ).to_csv(f"artifacts/metrics/t006_fullscale_analysis/fulltext_high_freq_trigrams_{args.output_prefix}.csv", index=False)
    pd.DataFrame([{"abbreviation": k, "count": v} for k, v in abbr_counter.most_common(80)]).to_csv(
        f"artifacts/metrics/t006_fullscale_analysis/fulltext_abbreviation_top80_{args.output_prefix}.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "concept": "SKIPPED_FULLSCALE_CONCEPT_VARIANTS_LIGHT_LEXICAL_PASS",
                "total_mentions": "",
                "note": "概念变体全量正则扫描在轻量 lexical pass 中跳过；本轮报告保留 3028 样本观察作为语义说明。",
            }
        ]
    ).to_csv(
        f"artifacts/metrics/t006_fullscale_analysis/fulltext_concept_variants_summary_{args.output_prefix}.csv", index=False
    )
    pd.DataFrame(
        [{"marker": marker, "count": int(structured_marker_counter[marker]), "rate": mod.safe_rate(int(structured_marker_counter[marker]), parsed_success)} for marker in mod.STRUCTURED_MARKERS]
    ).to_csv(f"artifacts/metrics/t006_fullscale_analysis/structured_abstract_markers_{args.output_prefix}.csv", index=False)

    fig_dir = Path("reports/figures")
    mod.save_hist(rows_df["title_abstract_token_len"].tolist(), fig_dir / f"title_abstract_token_length_hist_{args.output_prefix}.png", "Title + abstract token length")
    mod.save_box(rows_df["title_abstract_token_len"].tolist(), fig_dir / f"title_abstract_token_length_box_{args.output_prefix}.png", "Title + abstract token length")
    mod.save_hist(rows_df["full_token_len"].tolist(), fig_dir / f"full_text_token_length_hist_{args.output_prefix}.png", "Full text token length")
    mod.save_box(rows_df["full_token_len"].tolist(), fig_dir / f"full_text_token_length_box_{args.output_prefix}.png", "Full text token length")

    q33 = rows_df["full_token_len"].quantile(1 / 3)
    q66 = rows_df["full_token_len"].quantile(2 / 3)
    rows_df["full_length_group"] = rows_df["full_token_len"].apply(lambda x: "short" if x <= q33 else ("medium" if x <= q66 else "long"))
    sample_df = pd.concat(
        [
            rows_df[rows_df["full_length_group"] == group].sample(
                n=min(args.sample_per_group, int((rows_df["full_length_group"] == group).sum())),
                random_state=42,
            )
            for group in ["short", "medium", "long"]
        ],
        ignore_index=True,
    )
    sample_df.to_csv(f"artifacts/metrics/t006_fullscale_analysis/fulltext_domain_sample_summary_{args.output_prefix}.csv", index=False)
    source_to_path = {str(p.relative_to(data_dir)): p for p in xml_files}
    sample_lines = [
        "# 15w 全文分层抽样阅读包",
        "",
        f"- 分层依据：`full_token_len` 三分位；short <= {int(q33)}，medium <= {int(q66)}，long > {int(q66)}。",
        f"- 每层抽样：{args.sample_per_group} 篇；共 {len(sample_df)} 篇。",
        "- 本文件只保存轻量预览，不保存全量正文。",
        "",
    ]
    for _, row in sample_df.iterrows():
        _, text_full, _ = parse_text_only(source_to_path[str(row["source_file"])], mod)
        sample_lines.extend(
            [
                f"## {row['full_length_group']} | {row['record_id']} | {int(row['full_token_len'])} tokens",
                f"- title: {row['title']}",
                f"- journal/year: {row['journal']} / {row['pub_year']}",
                f"- pmid/pmcid: {row['pmid']} / {row['pmcid']}",
                f"- source_file: {row['source_file']}",
                f"- strategy: {row['recommended_split_strategy_cn']} (`{row['recommended_split_strategy']}`)",
                "",
                mod.preview_text(text_full),
                "",
            ]
        )
    Path(f"reports/samples/fulltext_stratified_sample_for_review_{args.output_prefix}.md").write_text("\n".join(sample_lines), encoding="utf-8")

    disk = mod.disk_usage_row(project_dir)
    elapsed = time.perf_counter() - started
    pd.DataFrame(
        [
            {"metric": "processed_date", "value": datetime.now().isoformat(timespec="seconds")},
            {"metric": "input_xml", "value": len(xml_files)},
            {"metric": "parsed_success", "value": parsed_success},
            {"metric": "parsed_failed", "value": parsed_failed},
            {"metric": "resume_existing_rows", "value": existing_count},
            {"metric": "elapsed_seconds_resume_and_finalize", "value": f"{elapsed:.3f}"},
            {"metric": "output_prefix", "value": args.output_prefix},
            {"metric": "light_table", "value": str(light_path)},
            {"metric": "report", "value": f"reports/formal/RAG数据分析与设计说明_{args.output_prefix}.md"},
            {"metric": "chroma_created", "value": "false"},
            {"metric": "chunk_dataset_created", "value": "false"},
            {"metric": "pdf_created", "value": "false"},
            *[{"metric": f"disk_{k}", "value": v} for k, v in disk.items()],
        ]
    ).to_csv(f"artifacts/metrics/t006_fullscale_analysis/fullscale_analysis_summary_{args.output_prefix}.csv", index=False)

    missing_lookup = {row["field"]: row for row in missing_rows}
    token_lookup = {row["text_field"]: row for row in token_stats}
    ta = token_lookup["text_title_abstract"]
    full = token_lookup["text_full"]
    route_rows = [
        {
            "strategy": mod.strategy_label(route),
            "route": route,
            "records": int(count),
            "rate": mod.rate(int(count), parsed_success),
            "estimated_chunks_if_implemented": int(route_est_chunks[route]),
        }
        for route, count in sorted(route_counts.items())
    ]
    top_abbr = [k for k, _ in abbr_counter.most_common(12)]
    top_uni = [" ".join(k) for k, _ in unigram_counter.most_common(12)]
    top_bi = ["轻量版跳过全量 bigram/trigram"]

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

## 4. Token 长度分布

| 文本 | count | mean | median | p95 | p99 | >512 数 | >512 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| title + abstract | {ta['count']} | {ta['mean']:.2f} | {ta['median']:.2f} | {ta['p95']:.2f} | {ta['p99']:.2f} | {ta['over_512_count']} | {ta['over_512_rate']:.2%} |
| full text | {full['count']} | {full['mean']:.2f} | {full['median']:.2f} | {full['p95']:.2f} | {full['p99']:.2f} | {full['over_512_count']} | {full['over_512_rate']:.2%} |

结论：摘要文本存在长尾，全文通常需要切分或章节化处理。长度图见 `reports/figures/*_{args.output_prefix}.png`。

## 5. 领域语言与结构特点

- 任意正文 section title 覆盖率：`{rows_df['has_any_section_title'].mean():.2%}`
- IMRaD core 覆盖率：`{rows_df['imrad_core'].mean():.2%}`
- 含 Conclusion/Summary 的 IMRaD 覆盖率：`{rows_df['imrad_with_conclusion'].mean():.2%}`
- 全文高频缩写：`{", ".join(top_abbr)}`
- 高频 unigram：`{", ".join(top_uni)}`
- 高频 bigram/trigram：`{", ".join(top_bi)}`

医学文本存在大量缩写、全称、同义表达和统计符号。后续 prompt/query rewrite 应保留原始术语，同时支持常见缩写与全称互扩展。本轮为避免 15w 全文 bigram/trigram Counter 长尾导致运行过慢，领域语言部分采用全量 unigram、全量缩写和全量章节结构统计；bigram/trigram 不作为本轮正式全量指标。

## 6. 文本分割策略

mentor 给出的三类策略不是全局三选一，而是按文献长度和结构进行条件路由。本轮建议：

{mod.markdown_table(route_rows, ['strategy', 'route', 'records', 'rate', 'estimated_chunks_if_implemented'])}

策略解释：

- **整体不分割**：全文不超过 `{mod.WHOLE_DOC_TOKEN_LIMIT}` tokens 时，保留完整上下文。
- **按语义章节分割**：正文有 XML section title 时，优先按章节保留 Background/Methods/Results/Discussion/Conclusion 等结构；超长章节在下周实际 chunk 阶段再使用 recursive split。
- **重叠滑动窗口兜底**：无明确章节但全文较长时，使用 `RecursiveCharacterTextSplitter`，建议 `chunk_size={mod.CHUNK_SIZE}`、`chunk_overlap={mod.CHUNK_OVERLAP}`。

本轮只输出策略统计，不生成实际 chunk 数据集。下周任务再保存 `chunk_id/text/doc_id/chunk_index/total_chunks/source_title/token_count` 等字段。

## 7. 关键产物

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

## 8. 验证结论

- XML 总数验证：`{len(xml_files)}`。
- 解析计数验证：`parsed_success + parsed_failed = {parsed_success + parsed_failed}`。
- 本轮未生成 Chroma、未生成 chunk dataset、未生成 PDF。
- 本次使用已有 `{existing_count}` 条轻量表续跑，没有丢弃前序进度。
"""
    report_path = Path(f"reports/formal/RAG数据分析与设计说明_{args.output_prefix}.md")
    mod.write_markdown(report_path, "RAG数据分析与设计说明（15w PMC OA）", report_body)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"RESUME END {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Final parsed_success={parsed_success}; parsed_failed={parsed_failed}; report={report_path}\n")


if __name__ == "__main__":
    main()
