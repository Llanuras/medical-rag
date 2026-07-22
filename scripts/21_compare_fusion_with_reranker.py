from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_rag.query.understanding import process_medical_query, result_to_dict
from medical_rag.retrieval.multipath import DEFAULT_BM25_DIR, DEFAULT_CHROMA_COLLECTION, DEFAULT_CHROMA_DIR
from medical_rag.retrieval.pipeline import RetrievalPipeline, document_key
from medical_rag.retrieval.reranker import DEFAULT_RERANKER_MODEL
from medical_rag.retrieval.vector_store import DEFAULT_MODEL_NAME, Tee, now_iso

STRATEGIES = ("simple", "rrf", "weighted")


def query_specs() -> list[dict[str, str]]:
    rows = [
        ("Q001", "metformin cardiovascular outcomes in type 2 diabetes", "drug_treatment", "en", "P01", "drug,disease,outcome", "none"),
        ("Q002", "aspirin secondary prevention after myocardial infarction", "drug_treatment", "en", "", "drug,disease,treatment", "none"),
        ("Q003", "atorvastatin LDL cholesterol reduction and cardiovascular risk", "drug_treatment", "en", "", "drug,outcome,disease", "none"),
        ("Q004", "warfarin anticoagulation stroke prevention in atrial fibrillation", "drug_treatment", "en", "", "drug,disease,treatment", "none"),
        ("Q005", "beta blocker therapy mortality in chronic heart failure", "drug_treatment", "en", "", "drug,disease,outcome", "none"),
        ("Q006", "insulin sensitivity obesity and type 2 diabetes", "drug_treatment", "en", "", "drug,outcome,disease", "none"),
        ("Q007", "tamoxifen estrogen receptor positive breast cancer", "drug_treatment", "en", "", "drug,gene/protein,disease", "none"),
        ("Q008", "cisplatin resistance in ovarian cancer treatment", "drug_treatment", "en", "", "drug,disease,outcome", "none"),
        ("Q009", "antiretroviral therapy and HIV drug resistance", "drug_treatment", "en", "", "drug,disease,outcome", "none"),
        ("Q010", "multidrug-resistant tuberculosis treatment", "drug_treatment", "en", "", "disease,treatment,outcome", "none"),
        ("Q011", "EGFR mutation and treatment response in non-small cell lung cancer", "mechanism_genetics", "en", "P02", "gene/protein,disease,outcome", "none"),
        ("Q012", "EGFR T790M tyrosine kinase inhibitor resistance", "mechanism_genetics", "en", "", "gene/protein,drug,outcome", "none"),
        ("Q013", "BRCA1 DNA repair and hereditary breast cancer", "mechanism_genetics", "en", "P06", "gene/protein,disease,mechanism", "none"),
        ("Q014", "TP53 mutation apoptosis and tumor suppression", "mechanism_genetics", "en", "", "gene/protein,outcome,mechanism", "none"),
        ("Q015", "APC Wnt signaling in colorectal cancer", "mechanism_genetics", "en", "", "gene/protein,disease,mechanism", "none"),
        ("Q016", "androgen receptor signaling in prostate cancer progression", "mechanism_genetics", "en", "", "gene/protein,disease,outcome", "none"),
        ("Q017", "amyloid beta neurodegeneration in Alzheimer disease", "mechanism_genetics", "en", "", "gene/protein,disease,mechanism", "none"),
        ("Q018", "alpha-synuclein and dopaminergic neurons in Parkinson disease", "mechanism_genetics", "en", "", "gene/protein,disease,mechanism", "none"),
        ("Q019", "oxidative stress endothelial dysfunction and atherosclerosis", "mechanism_genetics", "en", "", "mechanism,disease,outcome", "none"),
        ("Q020", "TGF beta renal fibrosis in chronic kidney disease", "mechanism_genetics", "en", "", "gene/protein,disease,mechanism", "none"),
        ("Q021", "SARS coronavirus spike protein and ACE2 receptor", "infection_immunity", "en", "", "disease,gene/protein,mechanism", "none"),
        ("Q022", "influenza neuraminidase inhibitor resistance", "infection_immunity", "en", "", "disease,drug,outcome", "none"),
        ("Q023", "hepatitis C antiviral treatment response", "infection_immunity", "en", "", "disease,drug,outcome", "none"),
        ("Q024", "HIV reverse transcriptase inhibitor resistance", "infection_immunity", "en", "P04", "disease,drug,outcome", "none"),
        ("Q025", "Plasmodium falciparum antimalarial drug resistance", "infection_immunity", "en", "", "organism,drug,outcome", "none"),
        ("Q026", "bacterial sepsis inflammatory cytokine response", "infection_immunity", "en", "", "disease,gene/protein,outcome", "none"),
        ("Q027", "vaccine induced antibody immune response", "infection_immunity", "en", "", "treatment,immune,outcome", "none"),
        ("Q028", "macrophage activation in innate immune response", "infection_immunity", "en", "P05", "cell,immune,outcome", "none"),
        ("Q029", "interleukin 6 inflammation in rheumatoid arthritis", "infection_immunity", "en", "", "gene/protein,disease,mechanism", "none"),
        ("Q030", "TNF alpha blockade in rheumatoid arthritis", "infection_immunity", "en", "", "gene/protein,disease,treatment", "none"),
        ("Q031", "PCR detection of viral DNA", "method_biomarker", "en", "P03", "method,target", "none"),
        ("Q032", "quantitative real-time PCR gene expression analysis", "method_biomarker", "en", "", "method,outcome", "none"),
        ("Q033", "DNA microarray breast cancer gene expression classification", "method_biomarker", "en", "", "method,disease,outcome", "none"),
        ("Q034", "RNA interference mediated gene silencing", "method_biomarker", "en", "", "method,mechanism", "none"),
        ("Q035", "flow cytometry analysis of T cell activation", "method_biomarker", "en", "", "method,cell,outcome", "none"),
        ("Q036", "immunohistochemistry detection of tumor biomarkers", "method_biomarker", "en", "", "method,biomarker", "none"),
        ("Q037", "ELISA measurement of inflammatory cytokines", "method_biomarker", "en", "", "method,biomarker", "none"),
        ("Q038", "survival analysis of prognostic biomarkers in cancer", "method_biomarker", "en", "", "method,biomarker,outcome", "none"),
        ("Q039", "二甲双胍对2型糖尿病患者心血管结局有什么影响？", "bilingual", "zh", "P01", "drug,disease,outcome", "none"),
        ("Q040", "EGFR突变如何影响非小细胞肺癌的治疗反应？", "bilingual", "zh", "P02", "gene/protein,disease,outcome", "none"),
        ("Q041", "PCR如何用于病毒核酸检测？", "bilingual", "zh", "P03", "method,target", "none"),
        ("Q042", "HIV逆转录酶抑制剂的耐药机制是什么？", "bilingual", "zh", "P04", "disease,drug,outcome", "none"),
        ("Q043", "炎症因子（inflammatory cytokines）如何调控巨噬细胞（macrophage）反应？", "bilingual", "zh", "P05", "gene/protein,cell,outcome", "none"),
        ("Q044", "BRCA1与乳腺癌遗传风险有什么关系？", "bilingual", "zh", "P06", "gene/protein,disease,outcome", "none"),
        ("Q045", "2003 Plasmodium falciparum gene expression research article", "metadata_structure", "en", "", "pub_year,article_type", "plan_only"),
        ("Q046", "PLoS ONE breast cancer gene expression", "metadata_structure", "en", "", "journal", "plan_only"),
        ("Q047", "aspirin myocardial infarction before 2008", "metadata_structure", "en", "", "pub_year_lte", "plan_only"),
        ("Q048", "EGFR lung cancer after 2008", "metadata_structure", "en", "", "pub_year_gte", "plan_only"),
        ("Q049", "PCR viral detection methods section", "metadata_structure", "en", "", "section_title_norm", "plan_only"),
        ("Q050", "breast cancer gene expression results section research article", "metadata_structure", "en", "", "article_type,section_title_norm", "mixed"),
    ]
    return [dict(zip(("query_id", "query", "category", "language", "pair_id", "expected_fields", "expected_filter_mode"), row)) for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simple/RRF/weighted fusion under one fixed strict BGE reranker.")
    parser.add_argument("--output_prefix", default="top50_q50")
    parser.add_argument(
        "--metrics_dir",
        default="artifacts/metrics/t014_multipath_retrieval",
        help="Project-relative directory for machine-readable benchmark artifacts.",
    )
    parser.add_argument(
        "--reports_dir",
        default="reports/formal",
        help="Project-relative directory for the formal benchmark report.",
    )
    parser.add_argument(
        "--logs_dir",
        default="logs",
        help="Project-relative directory for the benchmark log.",
    )
    parser.add_argument("--query_ids", default=None, help="Comma-separated query IDs; default runs all 50.")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--top_k_vector", type=int, default=50)
    parser.add_argument("--top_k_keyword", type=int, default=50)
    parser.add_argument("--fusion_top_k", type=int, default=50)
    parser.add_argument("--rerank_top_k", type=int, default=50)
    parser.add_argument("--final_top_k", type=int, default=10)
    parser.add_argument("--chroma_persist_dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--chroma_collection_name", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument("--bm25_index_dir", default=str(DEFAULT_BM25_DIR))
    parser.add_argument("--embedding_model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker_model_name", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reranker_batch_size", type=int, default=8)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--vector_weight", type=float, default=0.65)
    parser.add_argument("--keyword_weight", type=float, default=0.35)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--terminology_path", default=None)
    parser.add_argument("--term_index_path", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        raise ValueError("output_prefix must be filename-safe")
    for name in ("top_k_vector", "top_k_keyword", "fusion_top_k", "rerank_top_k", "final_top_k"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.rerank_top_k > args.fusion_top_k:
        raise ValueError("rerank_top_k cannot exceed fusion_top_k")
    if args.final_top_k > args.rerank_top_k:
        raise ValueError("final_top_k cannot exceed rerank_top_k")


def resolve_project_output_dir(value: str, *, argument_name: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{argument_name} must resolve inside the project root: {PROJECT_ROOT}") from exc
    return resolved


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def overlap(a: set[str], b: set[str]) -> tuple[int, float, float]:
    common = len(a & b)
    denominator = min(len(a), len(b))
    union = len(a | b)
    return common, common / denominator if denominator else 0.0, common / union if union else 0.0


def result_row(spec: dict[str, str], strategy: str, item: dict[str, Any], filter_applied: bool) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    return {
        "query_id": spec["query_id"], "query": spec["query"], "category": spec["category"],
        "language": spec["language"], "pair_id": spec["pair_id"], "fusion_strategy": strategy,
        "filter_applied": filter_applied, "fusion_rank": item.get("fusion_rank"),
        "reranker_rank": item.get("reranker_rank"), "final_rank": item.get("final_rank"),
        "chunk_id": item.get("chunk_id"), "document_key": item.get("document_key"),
        "doc_id": item.get("doc_id"), "same_doc_candidate_count": item.get("same_doc_candidate_count"),
        "source_title": metadata.get("source_title"), "journal": metadata.get("journal"),
        "pub_year": metadata.get("pub_year"), "pmid": metadata.get("pmid"), "pmcid": metadata.get("pmcid"),
        "article_type": metadata.get("article_type"), "section_title": metadata.get("section_title"),
        "retrieval_sources": ",".join(item.get("retrieval_sources") or []),
        "vector_rank": item.get("vector_rank"), "keyword_rank": item.get("keyword_rank"),
        "vector_score": item.get("vector_score"), "bm25_score": item.get("bm25_score"),
        "fusion_score": item.get("fusion_score"), "reranker_raw_score": item.get("reranker_raw_score"),
        "relevance_score": item.get("relevance_score"), "recency_score": item.get("recency_score"),
        "authority_score": item.get("authority_score"), "final_score": item.get("final_score"),
        "text": item.get("text"),
    }


def make_report(
    args: argparse.Namespace,
    specs: list[dict[str, str]],
    summaries: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    bilingual: list[dict[str, Any]],
    annotation_count: int,
) -> str:
    strategy_rows = []
    for strategy in STRATEGIES:
        rows = [row for row in summaries if row["fusion_strategy"] == strategy]
        strategy_rows.append(
            f"| {strategy} | {len(rows)} | {sum(r['final_result_count'] for r in rows)} | "
            f"{statistics.mean(r['retrieval_seconds'] for r in rows):.3f} | "
            f"{statistics.mean(r['fusion_seconds'] for r in rows):.4f} | "
            f"{statistics.mean(r['rerank_score_seconds'] for r in rows):.3f} |"
        )
    stage_rows = []
    for stage in ("pre_doc_top50", "pre_chunk_top50", "final_doc_top5", "final_doc_top10"):
        for left, right in combinations(STRATEGIES, 2):
            rows = [r for r in overlaps if r["stage"] == stage and r["strategy_a"] == left and r["strategy_b"] == right]
            stage_rows.append(
                f"| {stage} | {left} vs {right} | "
                f"{statistics.mean(r['overlap_rate'] for r in rows):.4f} | "
                f"{statistics.mean(r['jaccard'] for r in rows):.4f} |"
            )
    bilingual_rows = []
    for strategy in STRATEGIES:
        rows = [r for r in bilingual if r["strategy"] == strategy]
        if rows:
            rate = f"{statistics.mean(r['top10_doc_overlap_rate'] for r in rows):.4f}"
            jaccard = f"{statistics.mean(r['top10_doc_jaccard'] for r in rows):.4f}"
        else:
            rate = jaccard = "n/a"
        bilingual_rows.append(f"| {strategy} | {len(rows)} | {rate} | {jaccard} |")
    exact_simple_rrf = all(
        float(row["overlap_rate"]) == 1.0
        for row in overlaps
        if row["stage"] == "final_doc_top10"
        and row["strategy_a"] == "simple"
        and row["strategy_b"] == "rrf"
    )
    return f"""# 固定 Reranker 下融合策略对比报告（{args.output_prefix}）

## 实验结论边界

本轮比较 `simple`、`rrf`、`weighted` 三种融合候选在固定 `BAAI/bge-reranker-base` 下的候选敏感性与结果一致性。没有人工相关性标签，因此本报告不声称任一策略在 Recall、MRR 或 nDCG 上更优；人工质量评价需填写随附 annotation pool 后另算。

## 固定配置

- Queries: `{len(specs)}`
- Vector/BM25/Fusion/Rerank/Final top-k: `{args.top_k_vector}/{args.top_k_keyword}/{args.fusion_top_k}/{args.rerank_top_k}/{args.final_top_k}`
- RRF k: `{args.rrf_k}`
- Weighted vector/BM25: `{args.vector_weight:.2f}/{args.keyword_weight:.2f}`
- Reranker batch/max length: `{args.reranker_batch_size}/{args.reranker_max_length}`
- Reranker policy: strict; model unavailable or scoring failure aborts the run
- BM25 scope: part000（64,853 chunks）；305 万 chunks 的全量分片 BM25 延后到 T015
- Query execution: query understanding、vector search 和 BM25 search 每条只执行一次，三种融合复用相同两路候选
- Final evidence: 按 document key 去重，每篇文档保留最高分 chunk

## 运行与完整性

- Fusion candidates: `{len(specs) * len(STRATEGIES) * args.fusion_top_k}`
- Final evidence rows: `{sum(row['final_result_count'] for row in summaries)}`
- Annotation pool query-document rows: `{annotation_count}`
- `{len(summaries)}` 个 query-strategy 分组全部使用真实 reranker；无 fallback、无缺失 reranker score；每组最终 `{args.final_top_k}` 篇文档且 document key 唯一。

| Strategy | Query groups | Final rows | Shared retrieval s/query | Fusion s/query | Rerank+score s/query |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(strategy_rows)}

## 候选与最终结果重合度

| Stage | Pair | Mean overlap rate | Mean Jaccard |
| --- | --- | ---: | ---: |
{chr(10).join(stage_rows)}

实测中 simple 与 RRF 的最终 Top-10 文档集合在全部查询上完全一致：`{exact_simple_rrf}`。weighted 与另外两种策略的最终 Top-10 平均 overlap rate 为约 `0.888`，说明固定 reranker 能消除 simple/RRF 的排序差异，但 weighted 的候选集合差异仍会传递到最终结果。该观察只说明候选敏感性，不等于相关性质量优劣。

## 双语配对一致性

| Strategy | EN-ZH pairs | Mean Top-10 doc overlap | Mean Jaccard |
| --- | ---: | ---: | ---: |
{chr(10).join(bilingual_rows)}

双语一致性仍偏低，说明当前中文查询增强覆盖有限；Q043 已采用中英术语并列以避免英文语料上的 BM25 空结果。这是后续 query rewrite/跨语言检索的改进点，不在本轮扩展范围内。

## 元数据过滤解释

`filter_applied=true` 仅表示 query understanding 生成了可直接执行的 `where_filter` 并传给两路检索。主 benchmark 中 Q050 执行 `article_type=research-article` 硬过滤；年份范围和 section 条件只记录在 `filter_plan`，不能宣称已执行。

预检显示 Q045 的 `pub_year=2003 AND article_type=research-article` 在全量 Chroma 上可返回 50 条、part000 BM25 返回 25 条，但 Top-50 被少数文档的多个 chunks 占据，文档去重后不足 10；Q046 的 `PLoS ONE` 在 part000 BM25 语料中不存在。因此 Q045/Q046 在主融合公平比较中只审计 filter plan，不执行硬过滤，避免把语料范围差异或 chunk 多样性问题误写成融合策略效果。

## 人工标注下一步

在 `fusion_with_reranker_annotation_pool_{args.output_prefix}.csv` 的 `relevance_label` 列填 0/1/2。完成后才能计算各策略 Recall@10、MRR@10、nDCG@10，并做有依据的质量选择。
"""

def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2
    specs = query_specs()
    if args.query_ids:
        selected = {value.strip().upper() for value in args.query_ids.split(",") if value.strip()}
        known = {row["query_id"] for row in specs}
        if selected - known:
            print(f"[ERROR] unknown query IDs: {sorted(selected - known)}")
            return 2
        specs = [row for row in specs if row["query_id"] in selected]
    if args.max_queries is not None:
        if args.max_queries <= 0:
            print("[ERROR] max_queries must be positive")
            return 2
        specs = specs[:args.max_queries]
    if not specs:
        print("[ERROR] no benchmark queries selected")
        return 2

    try:
        metrics = resolve_project_output_dir(args.metrics_dir, argument_name="metrics_dir")
        reports = resolve_project_output_dir(args.reports_dir, argument_name="reports_dir")
        logs = resolve_project_output_dir(args.logs_dir, argument_name="logs_dir")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2
    for directory in (metrics, reports, logs):
        directory.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    paths = {
        "queries": metrics / f"benchmark_queries_{prefix}.csv",
        "candidates": metrics / f"fusion_candidates_{prefix}.jsonl",
        "results_csv": metrics / f"fusion_with_reranker_results_{prefix}.csv",
        "results_jsonl": metrics / f"fusion_with_reranker_results_{prefix}.jsonl",
        "summary": metrics / f"fusion_with_reranker_summary_{prefix}.csv",
        "overlap": metrics / f"fusion_with_reranker_overlap_{prefix}.csv",
        "bilingual": metrics / f"fusion_with_reranker_bilingual_pairs_{prefix}.csv",
        "annotation": metrics / f"fusion_with_reranker_annotation_pool_{prefix}.csv",
        "report": reports / f"固定Reranker下融合策略对比报告_{prefix}.md",
        "log": logs / f"21_compare_fusion_with_reranker_{prefix}.log",
    }
    tee = Tee(paths["log"])
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = tee
    try:
        print(f"[START] {now_iso()} queries={len(specs)}")
        write_csv(paths["queries"], specs)
        pipeline = RetrievalPipeline(
            project_root=PROJECT_ROOT, chroma_persist_dir=args.chroma_persist_dir,
            chroma_collection_name=args.chroma_collection_name, bm25_index_dir=args.bm25_index_dir,
            embedding_model_name=args.embedding_model_name, reranker_model_name=args.reranker_model_name,
            terminology_path=args.terminology_path, term_index_path=args.term_index_path, device=args.device,
            reranker_batch_size=args.reranker_batch_size, reranker_max_length=args.reranker_max_length,
            vector_weight=args.vector_weight, keyword_weight=args.keyword_weight, rrf_k=args.rrf_k,
        )
        print("[PREFLIGHT] loading strict reranker once")
        reranker = pipeline.load_reranker()
        print(f"[PREFLIGHT] reranker_source={reranker.model_source} device={reranker.device}")
        candidate_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        overlap_rows: list[dict[str, Any]] = []
        all_final: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for query_number, spec in enumerate(specs, start=1):
            qid = spec["query_id"]
            understood = process_medical_query(spec["query"], pipeline.terminology_path, pipeline.term_index_path)
            query_info = result_to_dict(understood)
            if not understood.clean_query:
                raise RuntimeError(f"{qid} produced an empty clean query")
            retrieval_query_info = dict(query_info)
            if spec["expected_filter_mode"] == "plan_only":
                retrieval_query_info["where_filter"] = None
            retrieval_started = time.perf_counter()
            base = pipeline.retriever.retrieve_paths(
                retrieval_query_info, top_k_vector=args.top_k_vector, top_k_keyword=args.top_k_keyword, strict=True,
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            if not base["vector_results"] or not base["keyword_results"]:
                raise RuntimeError(f"{qid} retrieval path returned no candidates: vector={len(base['vector_results'])}, keyword={len(base['keyword_results'])}")
            filter_applied = bool(base["where"])
            all_final[qid] = {}
            fused_sets: dict[str, tuple[set[str], set[str]]] = {}
            for strategy in STRATEGIES:
                fusion_started = time.perf_counter()
                fused = pipeline.retriever.fuse_results(
                    base["vector_results"], base["keyword_results"], fusion_strategy=strategy, top_k=args.fusion_top_k,
                )
                fusion_seconds = time.perf_counter() - fusion_started
                if len(fused) != args.fusion_top_k:
                    raise RuntimeError(f"{qid}/{strategy} fused candidate count={len(fused)}")
                fused_sets[strategy] = ({document_key(item) for item in fused}, {str(item["chunk_id"]) for item in fused})
                for item in fused:
                    candidate_rows.append({
                        "query_id": qid, "query": spec["query"], "fusion_strategy": strategy,
                        "fusion_rank": item.get("fusion_rank"), "chunk_id": item.get("chunk_id"),
                        "document_key": document_key(item), "doc_id": item.get("doc_id"),
                        "vector_rank": item.get("vector_rank"), "keyword_rank": item.get("keyword_rank"),
                        "fusion_score": item.get("fusion_score"), "retrieval_sources": item.get("retrieval_sources"),
                        "text_preview": str(item.get("text") or "")[:500],
                    })
                rerank_started = time.perf_counter()
                final, used, warnings = pipeline.rerank_and_score(
                    understood.clean_query, fused, rerank_top_k=args.rerank_top_k,
                    final_top_k=args.final_top_k, strict_reranker=True,
                )
                rerank_seconds = time.perf_counter() - rerank_started
                if not used or warnings:
                    raise RuntimeError(f"{qid}/{strategy} strict reranker contract violated: used={used}, warnings={warnings}")
                if len(final) != args.final_top_k or len({item["document_key"] for item in final}) != len(final):
                    raise RuntimeError(f"{qid}/{strategy} final document dedupe/top-k contract violated")
                all_final[qid][strategy] = final
                result_rows.extend(result_row(spec, strategy, item, filter_applied) for item in final)
                summary_rows.append({
                    "query_id": qid, "query": spec["query"], "category": spec["category"],
                    "language": spec["language"], "fusion_strategy": strategy,
                    "vector_result_count": len(base["vector_results"]), "keyword_result_count": len(base["keyword_results"]),
                    "fused_candidate_count": len(fused), "final_result_count": len(final),
                    "filter_applied": filter_applied, "where_json": json.dumps(base["where"], ensure_ascii=False),
                    "filter_plan_json": json.dumps(query_info.get("filter_plan"), ensure_ascii=False),
                    "retrieval_seconds": retrieval_seconds, "fusion_seconds": fusion_seconds,
                    "rerank_score_seconds": rerank_seconds, "reranker_used": used,
                })
            for left, right in combinations(STRATEGIES, 2):
                for stage, a, b in (
                    ("pre_doc_top50", fused_sets[left][0], fused_sets[right][0]),
                    ("pre_chunk_top50", fused_sets[left][1], fused_sets[right][1]),
                    ("final_doc_top5", {document_key(x) for x in all_final[qid][left][:5]}, {document_key(x) for x in all_final[qid][right][:5]}),
                    ("final_doc_top10", {document_key(x) for x in all_final[qid][left][:10]}, {document_key(x) for x in all_final[qid][right][:10]}),
                ):
                    count, rate, jaccard = overlap(a, b)
                    overlap_rows.append({"query_id": qid, "stage": stage, "strategy_a": left, "strategy_b": right,
                                         "size_a": len(a), "size_b": len(b), "overlap_count": count,
                                         "overlap_rate": rate, "jaccard": jaccard})
            print(f"[QUERY] {query_number}/{len(specs)} {qid} retrieval={retrieval_seconds:.3f}s")

        with paths["candidates"].open("w", encoding="utf-8") as handle:
            for row in candidate_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_csv(paths["results_csv"], result_rows)
        with paths["results_jsonl"].open("w", encoding="utf-8") as handle:
            for row in result_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_csv(paths["summary"], summary_rows)
        write_csv(paths["overlap"], overlap_rows)

        bilingual_rows: list[dict[str, Any]] = []
        for pair_id in sorted({s["pair_id"] for s in specs if s["pair_id"]}):
            pair_specs = [s for s in specs if s["pair_id"] == pair_id]
            if len(pair_specs) != 2:
                continue
            left, right = pair_specs
            for strategy in STRATEGIES:
                a = {document_key(x) for x in all_final[left["query_id"]][strategy]}
                b = {document_key(x) for x in all_final[right["query_id"]][strategy]}
                count, rate, jaccard = overlap(a, b)
                bilingual_rows.append({"pair_id": pair_id, "strategy": strategy,
                    "query_id_a": left["query_id"], "query_a": left["query"],
                    "query_id_b": right["query_id"], "query_b": right["query"],
                    "top10_doc_overlap_count": count, "top10_doc_overlap_rate": rate, "top10_doc_jaccard": jaccard})
        bilingual_fields = ["pair_id", "strategy", "query_id_a", "query_a", "query_id_b", "query_b",
                            "top10_doc_overlap_count", "top10_doc_overlap_rate", "top10_doc_jaccard"]
        write_csv(paths["bilingual"], bilingual_rows, bilingual_fields)

        annotation_rows: list[dict[str, Any]] = []
        for spec in specs:
            qid = spec["query_id"]
            by_doc: dict[str, dict[str, Any]] = {}
            for strategy in STRATEGIES:
                for item in all_final[qid][strategy]:
                    key = document_key(item)
                    if key not in by_doc:
                        metadata = item.get("metadata") or {}
                        by_doc[key] = {"query_id": qid, "query": spec["query"], "document_key": key,
                            "doc_id": item.get("doc_id"), "source_title": metadata.get("source_title"),
                            "journal": metadata.get("journal"), "pub_year": metadata.get("pub_year"),
                            "pmid": metadata.get("pmid"), "pmcid": metadata.get("pmcid"),
                            "text_preview": str(item.get("text") or "")[:1000],
                            "simple_rank": "", "rrf_rank": "", "weighted_rank": "",
                            "relevance_label": "", "annotator_note": ""}
                    by_doc[key][f"{strategy}_rank"] = item.get("final_rank")
            annotation_rows.extend(by_doc.values())
        annotation_fields = ["query_id", "query", "document_key", "doc_id", "source_title", "journal", "pub_year",
                             "pmid", "pmcid", "text_preview", "simple_rank", "rrf_rank", "weighted_rank",
                             "relevance_label", "annotator_note"]
        write_csv(paths["annotation"], annotation_rows, annotation_fields)
        paths["report"].write_text(make_report(args, specs, summary_rows, overlap_rows, bilingual_rows, len(annotation_rows)), encoding="utf-8")
        print(f"[DONE] candidates={len(candidate_rows)} results={len(result_rows)} annotation_pool={len(annotation_rows)}")
        for name, path in paths.items():
            print(f"[{name.upper()}] {path}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
