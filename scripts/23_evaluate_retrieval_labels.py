from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


STRATEGIES = ("simple", "rrf", "weighted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pooled Recall@10, MRR@10, and nDCG@10 from judged retrieval results.")
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--label_column", default="relevance_label")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reports_dir", default="reports/formal")
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--relevant_threshold", type=int, default=2)
    parser.add_argument(
        "--label_source",
        choices=("human", "codex_reviewed", "llm_assisted"),
        default="human",
        help="Provenance recorded in the formal report; it does not alter metric formulas.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dcg(labels: list[int]) -> float:
    return sum((2**label - 1) / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def main() -> int:
    args = parse_args()
    if args.relevant_threshold not in {1, 2}:
        print("[ERROR] relevant_threshold must be 1 or 2")
        return 2
    results = read_csv(Path(args.results_csv))
    labels_rows = read_csv(Path(args.labels_csv))
    if not results or not labels_rows:
        print("[ERROR] results_csv and labels_csv must both contain rows")
        return 2
    label_by_query_doc: dict[tuple[str, str], int] = {}
    labels_by_query: dict[str, list[int]] = defaultdict(list)
    for row in labels_rows:
        key = (str(row.get("query_id") or ""), str(row.get("document_key") or ""))
        raw = str(row.get(args.label_column) or "").strip()
        if not all(key) or raw not in {"0", "1", "2"}:
            print(f"[ERROR] missing/invalid {args.label_column} for query-document pair: {key}")
            return 2
        if key in label_by_query_doc:
            print(f"[ERROR] duplicate label row: {key}")
            return 2
        label = int(raw)
        label_by_query_doc[key] = label
        labels_by_query[key[0]].append(label)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in results:
        grouped[(str(row.get("query_id") or ""), str(row.get("fusion_strategy") or ""))].append(row)
    if len(grouped) != 150:
        print(f"[ERROR] expected 150 query-strategy groups, got {len(grouped)}")
        return 2

    per_query: list[dict[str, object]] = []
    for (query_id, strategy), rows in sorted(grouped.items()):
        if strategy not in STRATEGIES or len(rows) != 10:
            print(f"[ERROR] invalid group {query_id}/{strategy}: rows={len(rows)}")
            return 2
        rows.sort(key=lambda row: int(row["final_rank"]))
        if [int(row["final_rank"]) for row in rows] != list(range(1, 11)):
            print(f"[ERROR] invalid final ranks for {query_id}/{strategy}")
            return 2
        retrieved_labels: list[int] = []
        for row in rows:
            key = (query_id, str(row.get("document_key") or ""))
            if key not in label_by_query_doc:
                print(f"[ERROR] unjudged retrieved result: {key}")
                return 2
            retrieved_labels.append(label_by_query_doc[key])
        pool_labels = labels_by_query[query_id]
        relevant_pool = sum(label >= args.relevant_threshold for label in pool_labels)
        retrieved_relevant = sum(label >= args.relevant_threshold for label in retrieved_labels)
        recall = retrieved_relevant / relevant_pool if relevant_pool else 0.0
        reciprocal_rank = next((1.0 / rank for rank, label in enumerate(retrieved_labels, start=1) if label >= args.relevant_threshold), 0.0)
        ideal = sorted(pool_labels, reverse=True)[:10]
        ideal_dcg = dcg(ideal)
        ndcg = dcg(retrieved_labels) / ideal_dcg if ideal_dcg else 0.0
        per_query.append(
            {
                "query_id": query_id,
                "fusion_strategy": strategy,
                "pooled_relevant_document_count": relevant_pool,
                "retrieved_relevant_count_at_10": retrieved_relevant,
                "pooled_recall_at_10": recall,
                "mrr_at_10": reciprocal_rank,
                "ndcg_at_10": ndcg,
            }
        )

    summary: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        rows = [row for row in per_query if row["fusion_strategy"] == strategy]
        summary.append(
            {
                "fusion_strategy": strategy,
                "query_count": len(rows),
                "pooled_recall_at_10_macro": mean([float(row["pooled_recall_at_10"]) for row in rows]),
                "mrr_at_10_macro": mean([float(row["mrr_at_10"]) for row in rows]),
                "ndcg_at_10_macro": mean([float(row["ndcg_at_10"]) for row in rows]),
            }
        )
    winner = max(summary, key=lambda row: (float(row["ndcg_at_10_macro"]), float(row["mrr_at_10_macro"]), float(row["pooled_recall_at_10_macro"])))

    output_dir = Path(args.output_dir)
    report_dir = Path(args.reports_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    if not report_dir.is_absolute():
        report_dir = Path.cwd() / report_dir
    per_query_path = output_dir / f"retrieval_label_metrics_per_query_{args.output_prefix}.csv"
    summary_path = output_dir / f"retrieval_label_metrics_summary_{args.output_prefix}.csv"
    report_path = report_dir / f"全量BM25融合策略相关性评估_{args.output_prefix}.md"
    write_csv(per_query_path, per_query)
    write_csv(summary_path, summary)
    table = "\n".join(
        f"| {row['fusion_strategy']} | {float(row['pooled_recall_at_10_macro']):.4f} | {float(row['mrr_at_10_macro']):.4f} | {float(row['ndcg_at_10_macro']):.4f} |"
        for row in summary
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# 全量 BM25 融合策略相关性评估（{args.output_prefix}）

## 评估边界

- 标签列：`{args.label_column}`
- 标签来源：`{args.label_source}`
- 相关阈值：`label >= {args.relevant_threshold}` 用于 pooled Recall@10 与 MRR@10。
- nDCG@10 使用 0/1/2 分级增益。
- pooled Recall@10 的分母是同一查询下标注池内所有相关文档，不是全语料的穷尽 Recall。

| Strategy | Pooled Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
{table}

## 当前选择

按 `nDCG@10` 主排序、`MRR@10` 和 pooled Recall@10 作为并列规则，当前最佳策略是：`{winner['fusion_strategy']}`。

{("该结果使用人工标签，可作为本轮正式相关性评估结论。" if args.label_source == "human" else "该结果不是人工金标准，仅可作为自动化或辅助筛选结论。")}
""",
        encoding="utf-8",
    )
    print(f"[DONE] winner={winner['fusion_strategy']} per_query={per_query_path} summary={summary_path} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
