# 全量 BM25 融合策略相关性评估（full_bm25_top50_q50_human_labels）

## 评估边界

- 标签列：`relevance_label`
- 标签来源：`human`
- 相关阈值：`label >= 2` 用于 pooled Recall@10 与 MRR@10。
- nDCG@10 使用 0/1/2 分级增益。
- pooled Recall@10 的分母是同一查询下标注池内所有相关文档，不是全语料的穷尽 Recall。

| Strategy | Pooled Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| simple | 0.8065 | 0.6866 | 0.8600 |
| rrf | 0.8127 | 0.6899 | 0.8614 |
| weighted | 0.8074 | 0.6640 | 0.8341 |

## 当前选择

按 `nDCG@10` 主排序、`MRR@10` 和 pooled Recall@10 作为并列规则，当前最佳策略是：`rrf`。

该结果使用人工标签，可作为本轮正式相关性评估结论。
