from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd

WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z\-]{2,}\b")
ABBR_RE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,12}\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

OUTPUT_SUBDIRS = [
    "artifacts/datasets/records",
    "artifacts/metrics/t002_corpus_analysis",
    "reports/figures",
    "reports/samples",
    "logs",
    "reports/technical",
]

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "has", "have", "had",
    "not", "but", "can", "may", "been", "their", "these", "those", "into", "using", "used", "between",
    "among", "also", "than", "then", "there", "such", "which", "when", "where", "during", "after", "before",
    "our", "all", "one", "two", "more", "most", "other", "over", "under", "within", "without", "each",
    "we", "they", "it", "its", "as", "in", "on", "of", "to", "a", "an", "by", "or", "is", "be", "at",
    "study", "studies", "results", "methods", "background", "conclusion", "conclusions", "objective", "objectives",
}

SECTION_PATTERNS = {
    "background_or_introduction": re.compile(r"\b(background|introduction)\b", re.I),
    "methods": re.compile(r"\b(methods?|materials and methods|methodology|patients and methods)\b", re.I),
    "results": re.compile(r"\bresults?\b", re.I),
    "discussion": re.compile(r"\bdiscussion\b", re.I),
    "conclusion": re.compile(r"\b(conclusions?|summary)\b", re.I),
}

NOISY_ABBR = {
    "PDF", "XML", "HTML", "FIG", "TABLE", "REF", "SUPPL", "PMC", "PMID", "DOI",
    "BMC", "PLOS", "USA", "UK", "WHO", "CDC",
    "C-", "T-", "N-", "L-", "P-", "S-", "I-", "M-",
}

NOISY_TERMS = STOPWORDS | {
    "article", "articles", "figure", "figures", "table", "tables", "supplementary",
    "additional", "file", "files", "copyright", "license", "published", "journal",
    "author", "authors", "contribution", "contributions", "competing", "interest",
    "interests", "available", "availability", "http", "https", "www", "com", "org",
    "result", "method", "background", "discussion", "conclusion", "introduction",
    "both", "only", "however", "shown", "number", "first", "including", "respectively",
    "therefore", "although", "according", "described", "reported", "observed", "whether",
    "because", "while", "could", "would", "should", "might", "must", "well", "same",
    "click", "here", "pre-publication", "history", "read", "approved", "final",
    "manuscript", "participated", "design", "carried",
}

NOISY_PHRASES = {
    "data shown", "click here", "pre-publication history", "approved final",
    "final manuscript", "read approved", "participated design", "carried out",
    "here data",
}

CASE_SENSITIVE_PATTERNS = {
    r"\bHIV(?:-1)?\b",
    r"\bAIDS\b",
    r"\bPCR\b",
    r"\bRT-PCR\b",
    r"\bCI\b",
    r"\b95%\s*CI\b",
    r"\bOR\b",
    r"\bCSF\b",
    r"\bMI\b",
    r"\bPCP\b",
    r"\bNSCLC\b",
    r"\bUC\b",
    r"\bIBD\b",
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
    "corticosteroids": [
        r"\bcorticosteroids?\b",
        r"\bsteroids?\b",
        r"\bglucocorticoids?\b",
    ],
}


def ensure_output_dirs(base: Path = Path(".")) -> None:
    for rel in OUTPUT_SUBDIRS:
        (base / rel).mkdir(parents=True, exist_ok=True)


def read_records_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def setup_tee(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    fh = log_path.open("w", encoding="utf-8")
    return fh, redirect_stdout(Tee(sys.stdout, fh)), redirect_stderr(Tee(sys.stderr, fh))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def filtered_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in NOISY_TERMS and len(w) >= 3]


def top_ngrams(tokens: list[str], n: int, limit: int = 50) -> list[dict]:
    counts: Counter[tuple[str, ...]] = Counter()
    for i in range(0, max(0, len(tokens) - n + 1)):
        gram = tuple(tokens[i:i + n])
        if any(t in NOISY_TERMS for t in gram):
            continue
        if " ".join(gram) in NOISY_PHRASES:
            continue
        counts[gram] += 1
    return [{"term": " ".join(k), "count": v} for k, v in counts.most_common(limit)]


def top_abbreviations(text: str, limit: int = 50) -> list[dict]:
    counts = Counter(a for a in ABBR_RE.findall(text or "") if a not in NOISY_ABBR and not a.isdigit())
    return [{"abbreviation": k, "count": v} for k, v in counts.most_common(limit)]


def sentence_stats(text: str) -> tuple[int, float]:
    sentences = [s.strip() for s in SENTENCE_RE.split(text or "") if len(s.strip()) >= 20]
    if not sentences:
        return 0, 0.0
    lengths = [len(words(s)) for s in sentences]
    return len(sentences), sum(lengths) / len(lengths)


def count_any(patterns: list[str], text: str) -> int:
    total = 0
    for pattern in patterns:
        flags = 0 if pattern in CASE_SENSITIVE_PATTERNS else re.I
        total += len(re.findall(pattern, text or "", flags=flags))
    return total


def variant_rows(text: str, scope: str) -> list[dict]:
    rows = []
    for concept, patterns in CONCEPT_VARIANTS.items():
        hits = {pattern: count_any([pattern], text) for pattern in patterns}
        nonzero = {pattern: count for pattern, count in hits.items() if count}
        rows.append({
            "scope": scope,
            "concept": concept,
            "variant_forms_found": len(nonzero),
            "total_mentions": sum(nonzero.values()),
            "forms": "; ".join(f"{pattern}={count}" for pattern, count in nonzero.items()),
        })
    return rows


def section_flags_from_existing(section_df: pd.DataFrame) -> pd.DataFrame:
    flags = section_df.copy()
    if "has_introduction" in flags.columns:
        flags["has_background_or_introduction"] = (
            flags["has_introduction"].astype(str).str.lower().eq("true")
        )
    for col in ["has_methods", "has_results", "has_discussion", "has_conclusion", "has_any_section_title"]:
        if col in flags.columns:
            flags[col] = flags[col].astype(str).str.lower().eq("true")
    flags["imrad_core"] = (
        flags.get("has_background_or_introduction", False)
        & flags.get("has_methods", False)
        & flags.get("has_results", False)
        & flags.get("has_discussion", False)
    )
    flags["imrad_with_conclusion"] = flags["imrad_core"] & flags.get("has_conclusion", False)
    return flags


def section_label(row: pd.Series) -> str:
    labels = []
    for col, label in [
        ("has_background_or_introduction", "B/I"),
        ("has_methods", "M"),
        ("has_results", "R"),
        ("has_discussion", "D"),
        ("has_conclusion", "C"),
    ]:
        if bool(row.get(col, False)):
            labels.append(label)
    return "+".join(labels) if labels else "no detected section titles"


def preview_text(text: str, max_chars: int = 2200) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--sections", required=True)
    parser.add_argument("--output_prefix", default="limit3028")
    parser.add_argument("--sample_per_group", type=int, default=8)
    args = parser.parse_args()

    ensure_output_dirs()
    log_path = Path(f"logs/04b_fulltext_domain_understanding_{args.output_prefix}.log")
    fh, out_ctx, err_ctx = setup_tee(log_path)
    with fh, out_ctx, err_ctx:
        records = load_jsonl(Path(args.records_jsonl))
        records_df = pd.DataFrame(records)
        tokens = read_records_csv(Path(args.tokens))
        sections = section_flags_from_existing(read_records_csv(Path(args.sections)))
        section_keep_cols = [
            "record_id",
            "source_file",
            "section_title_count",
            "has_any_section_title",
            "has_introduction",
            "has_methods",
            "has_results",
            "has_discussion",
            "has_conclusion",
            "has_background_or_introduction",
            "imrad_core",
            "imrad_with_conclusion",
        ]
        sections = sections[[c for c in section_keep_cols if c in sections.columns]]
        df = records_df.merge(tokens[["record_id", "full_token_len", "estimated_chunks_full"]], on="record_id")
        df = df.merge(sections, on=["record_id", "source_file"], how="left")
        df["full_token_len"] = pd.to_numeric(df["full_token_len"], errors="coerce").fillna(0).astype(int)
        df["estimated_chunks_full"] = pd.to_numeric(df["estimated_chunks_full"], errors="coerce").fillna(0).astype(int)

        q33 = df["full_token_len"].quantile(1 / 3)
        q66 = df["full_token_len"].quantile(2 / 3)

        def group(length: int) -> str:
            if length <= q33:
                return "short"
            if length <= q66:
                return "medium"
            return "long"

        df["full_length_group"] = df["full_token_len"].apply(group)
        samples = []
        for label in ["short", "medium", "long"]:
            group_df = df[df["full_length_group"] == label]
            samples.append(group_df.sample(n=min(args.sample_per_group, len(group_df)), random_state=42))
        sample_df = pd.concat(samples, ignore_index=True)
        sample_ids = set(sample_df["record_id"])
        sample_text = "\n\n".join(sample_df["text_full"].fillna("").astype(str).tolist())
        corpus_text = "\n\n".join(df["text_full"].fillna("").astype(str).tolist())

        sample_summary_rows = []
        full_sample_records = []
        for _, row in sample_df.iterrows():
            text = str(row.get("text_full", ""))
            toks = filtered_words(text)
            abbr_rows = top_abbreviations(text, 10)
            sent_count, avg_sentence_words = sentence_stats(text)
            word_count = len(words(text))
            abbr_count = sum(item["count"] for item in abbr_rows)
            stat_mentions = len(re.findall(r"\bp\s*[<=>]\s*0?\.\d+|\bCI\b|\bOR\b|\bRR\b|%", text, flags=re.I))
            sample_summary_rows.append({
                "full_length_group": row["full_length_group"],
                "record_id": row["record_id"],
                "pmcid": row.get("pmcid", ""),
                "pmid": row.get("pmid", ""),
                "title": row.get("title", ""),
                "journal": row.get("journal", ""),
                "pub_year": row.get("pub_year", ""),
                "full_token_len": row["full_token_len"],
                "estimated_chunks_full": row["estimated_chunks_full"],
                "section_signal": section_label(row),
                "imrad_core": bool(row.get("imrad_core", False)),
                "imrad_with_conclusion": bool(row.get("imrad_with_conclusion", False)),
                "word_count": word_count,
                "sentence_count": sent_count,
                "avg_sentence_words": round(avg_sentence_words, 2),
                "abbreviation_mentions_top10": abbr_count,
                "abbreviation_density_per_1000_words": round(abbr_count / max(1, word_count) * 1000, 2),
                "statistical_marker_mentions": stat_mentions,
                "top_abbreviations": "; ".join(f"{x['abbreviation']}({x['count']})" for x in abbr_rows),
                "top_terms": "; ".join(f"{x['term']}({x['count']})" for x in top_ngrams(toks, 1, 8)),
            })
            full_sample_records.append({
                "record_id": row["record_id"],
                "full_length_group": row["full_length_group"],
                "title": row.get("title", ""),
                "journal": row.get("journal", ""),
                "pub_year": row.get("pub_year", ""),
                "pmid": row.get("pmid", ""),
                "pmcid": row.get("pmcid", ""),
                "source_file": row.get("source_file", ""),
                "full_token_len": int(row["full_token_len"]),
                "text_full": text,
            })

        sample_summary = pd.DataFrame(sample_summary_rows).sort_values(["full_length_group", "full_token_len"])
        sample_summary_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_domain_sample_summary_{args.output_prefix}.csv")
        sample_summary.to_csv(sample_summary_path, index=False)

        full_sample_jsonl = Path(f"reports/samples/fulltext_stratified_sample_{args.output_prefix}.jsonl")
        write_jsonl(full_sample_records, full_sample_jsonl)

        all_terms = filtered_words(corpus_text)
        sample_terms = filtered_words(sample_text)
        unigram_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_high_freq_unigrams_{args.output_prefix}.csv")
        bigram_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_high_freq_bigrams_{args.output_prefix}.csv")
        trigram_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_high_freq_trigrams_{args.output_prefix}.csv")
        full_abbr_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_abbreviation_top50_{args.output_prefix}.csv")
        pd.DataFrame(top_ngrams(all_terms, 1, 80)).to_csv(unigram_path, index=False)
        pd.DataFrame(top_ngrams(all_terms, 2, 80)).to_csv(bigram_path, index=False)
        pd.DataFrame(top_ngrams(all_terms, 3, 80)).to_csv(trigram_path, index=False)
        pd.DataFrame(top_abbreviations(corpus_text, 80)).to_csv(full_abbr_path, index=False)

        variant_path = Path(f"artifacts/metrics/t002_corpus_analysis/fulltext_concept_variants_{args.output_prefix}.csv")
        pd.DataFrame(variant_rows(corpus_text, "corpus") + variant_rows(sample_text, "sample")).to_csv(variant_path, index=False)

        imrad_rate = df["imrad_core"].fillna(False).mean()
        imrad_conclusion_rate = df["imrad_with_conclusion"].fillna(False).mean()
        section_rate = df["has_any_section_title"].fillna(False).mean()
        sampled_imrad_rate = sample_summary["imrad_core"].mean()
        sampled_imrad_c_rate = sample_summary["imrad_with_conclusion"].mean()
        short_max = int(df[df.full_length_group == "short"]["full_token_len"].max())
        medium_min = int(df[df.full_length_group == "medium"]["full_token_len"].min())
        medium_max = int(df[df.full_length_group == "medium"]["full_token_len"].max())
        long_min = int(df[df.full_length_group == "long"]["full_token_len"].min())

        top_abbr = pd.read_csv(full_abbr_path).head(15)
        top_uni = pd.read_csv(unigram_path).head(15)
        top_bi = pd.read_csv(bigram_path).head(12)
        variant_df = pd.read_csv(variant_path)
        sample_variant_df = variant_df[(variant_df.scope == "sample") & (variant_df.variant_forms_found > 1)]
        group_stats = sample_summary.groupby("full_length_group").agg(
            n=("record_id", "count"),
            token_min=("full_token_len", "min"),
            token_max=("full_token_len", "max"),
            avg_sentence_words=("avg_sentence_words", "mean"),
            avg_abbr_density=("abbreviation_density_per_1000_words", "mean"),
            imrad_core_rate=("imrad_core", "mean"),
        ).reindex(["short", "medium", "long"])
        group_stats_md = "\n".join(
            [
                "| group | n | token range | avg sentence words | avg abbr / 1000 words | IMRaD core |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            + [
                (
                    f"| {idx} | {int(row.n)} | {int(row.token_min)}-{int(row.token_max)} | "
                    f"{row.avg_sentence_words:.1f} | {row.avg_abbr_density:.1f} | {row.imrad_core_rate:.1%} |"
                )
                for idx, row in group_stats.iterrows()
            ]
        )

        review_lines = [
            "# 全文分层抽样阅读包",
            "",
            f"- 分层依据：`full_token_len` 三分位；short <= {short_max}，medium {medium_min}-{medium_max}，long >= {long_min} tokens。",
            f"- 每层抽样：{args.sample_per_group} 篇；共 {len(sample_df)} 篇。",
            f"- 完整全文 JSONL：`reports/samples/fulltext_stratified_sample_{args.output_prefix}.jsonl`。",
            "",
        ]
        for _, row in sample_summary.iterrows():
            rec = sample_df[sample_df.record_id == row["record_id"]].iloc[0]
            review_lines.extend([
                f"## {row['full_length_group']} | {row['record_id']} | {row['full_token_len']} tokens",
                f"- title: {row['title']}",
                f"- journal/year: {row['journal']} / {row['pub_year']}",
                f"- pmid/pmcid: {row['pmid']} / {row['pmcid']}",
                f"- structure: {row['section_signal']} | IMRaD core={row['imrad_core']} | with conclusion={row['imrad_with_conclusion']}",
                f"- abbreviation density: {row['abbreviation_density_per_1000_words']} per 1000 words | top: {row['top_abbreviations']}",
                f"- top terms: {row['top_terms']}",
                "",
                "全文预览：",
                "",
                preview_text(str(rec.get("text_full", ""))),
                "",
            ])
        review_md_path = Path(f"reports/samples/fulltext_stratified_sample_for_review_{args.output_prefix}.md")
        review_md_path.write_text("\n".join(review_lines), encoding="utf-8")

        body = f"""
## 1. 任务口径

本报告按全文 `title + abstract + body` 完成领域内容理解，而不是只看摘要。分层依据为 `full_token_len`，每层抽样 `{args.sample_per_group}` 篇，共 `{len(sample_df)}` 篇全文；完整抽样全文保存在 `{full_sample_jsonl}`，人工阅读版保存在 `{review_md_path}`。

## 2. 全文分层抽样

- short：`<= {short_max}` tokens
- medium：`{medium_min}-{medium_max}` tokens
- long：`>= {long_min}` tokens

抽样明细见 `{sample_summary_path}`。短文多为评论、社论、通信、简短病例或无标准正文结构的文章；中长文更常出现结构化章节和方法/结果细节。

## 3. 结构：是否遵循 IMRaD

在 3028 篇全文中，检测到任意正文 section title 的比例为 `{section_rate:.1%}`；检测到 Introduction/Background + Methods + Results + Discussion 的 IMRaD core 比例为 `{imrad_rate:.1%}`；如果要求额外包含 Conclusion/Summary，则比例为 `{imrad_conclusion_rate:.1%}`。

在本次 `{len(sample_df)}` 篇全文样本中，IMRaD core 比例为 `{sampled_imrad_rate:.1%}`，包含 Conclusion/Summary 的比例为 `{sampled_imrad_c_rate:.1%}`。结论是：研究型原始论文较稳定遵循 IMRaD，但短文、社论、评论、软件说明、病例报告和部分 BMC 早期文章会使用 Background/Discussion/Summary、Case presentation、Availability and requirements 等非标准标题。后续 prompt 和评估规则不应硬编码只识别 Introduction/Methods/Results/Discussion 四个英文标题。

## 4. 术语与缩写密度

全文 top 缩写包括：`{", ".join(top_abbr.abbreviation.astype(str).head(12))}`。这些缩写覆盖分子生物学、感染病、统计学和临床研究场景，例如 DNA/RNA/PCR/IL/IFN/TNF 更偏基础与免疫机制，HIV/AIDS/CSF/PCP 偏疾病或临床对象，CI/OR/RR/SD 偏统计表达。

本语料明显存在“缩写 + 全称 + 同义词/近义词”并存的问题。样本中可见多种表述的概念包括：`{", ".join(sample_variant_df.concept.astype(str).head(8)) or "样本内未达到多表述阈值"}`。全集概念变体统计见 `{variant_path}`。后续 query 改写需要保留原词，同时扩展常见全称和缩写，例如 HIV/AIDS/human immunodeficiency virus、PCR/polymerase chain reaction、CSF/cerebrospinal fluid、PCP/Pneumocystis pneumonia。

## 5. 常用专业术语清单

- 分子/基因/细胞：DNA, RNA, PCR, RT-PCR, gene expression, amino acid, cell line, protein, transcription factor, SNP, GFP, BLAST, ATP。
- 免疫/感染/病毒：HIV, HIV-1, AIDS, IFN, TNF, IL, CD4, CD8, HSV, LPS, Pneumocystis pneumonia/PCP。
- 临床/疾病/患者：patients, treatment, control group, risk factors, breast cancer, prostate cancer, COPD, asthma, depression, psoriasis, myocardial infarction/MI。
- 统计/流行病学：CI, 95% CI, OR, RR, SD, ANOVA, statistically significant, sample size, standard deviation, prevalence, mortality。
- 公共卫生/医疗服务：health care, public health, United States, quality of life, data set, data analysis。

这些术语不应只按单词匹配处理。尤其是疾病、检测技术和统计效应量常以缩写、全称、连字符写法和复数形式同时出现。

## 6. 高频词与专业语言风格

全文高频单词 top 项包括：`{", ".join(top_uni.term.astype(str).head(12))}`。高频短语 top 项包括：`{", ".join(top_bi.term.astype(str).head(10))}`。

这些词说明该 3028 篇集合不是单一临床疾病语料，而是混合了临床研究、基础生物学、微生物/病毒、公共卫生、遗传与蛋白表达分析。医学文本信息密度高，常把研究对象、干预/暴露、检测技术、统计效应量、样本来源、时间和条件压缩在一句话中；一句话内常同时出现数字、括号、缩写、基因/蛋白名和统计符号。

本次全文样本的信息密度指标如下：

{group_stats_md}

## 7. 对提示词工程的启发

- 检索 query 改写：对缩写做双向扩展，但回答时保留原文术语和 PMCID/PMID。
- 结构化抽取：章节识别应支持 Background/Introduction、Materials and methods/Patients and methods、Results and discussion、Conclusion/Summary、Case presentation 等变体。
- 答案生成：要求模型区分研究背景、方法、主要结果、限制和结论，避免把 Discussion 中的推测写成实证结果。
- 评估基线：不要只用通顺度评估，应检查是否保留疾病/基因/药物/统计缩写、数值和研究对象限定条件。
- 全文 RAG：长文必须按章节优先切分；短文可以整体或低粒度切分，但仍需保留 article_type 和 section metadata。

## 8. 输出文件

- `{sample_summary_path}`
- `{review_md_path}`
- `{full_sample_jsonl}`
- `{full_abbr_path}`
- `{unigram_path}`
- `{bigram_path}`
- `{trigram_path}`
- `{variant_path}`
"""
        doc_path = Path(f"reports/technical/04b_fulltext_domain_understanding_{args.output_prefix}.md")
        write_markdown(doc_path, "医学全文领域内容理解报告", body)

        print("Wrote", sample_summary_path)
        print("Wrote", review_md_path)
        print("Wrote", full_sample_jsonl)
        print("Wrote", full_abbr_path)
        print("Wrote", unigram_path)
        print("Wrote", bigram_path)
        print("Wrote", trigram_path)
        print("Wrote", variant_path)
        print("Wrote", doc_path)
        print("Sample record ids:", ", ".join(sorted(sample_ids)))


if __name__ == "__main__":
    main()
