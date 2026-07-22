# 固定 Reranker 下融合策略对比报告（full_bm25_top50_q50）

## 实验结论边界

本轮比较 `simple`、`rrf`、`weighted` 三种融合候选在固定 `BAAI/bge-reranker-base` 下的候选敏感性与结果一致性。没有人工相关性标签，因此本报告不声称任一策略在 Recall、MRR 或 nDCG 上更优；人工质量评价需填写随附 annotation pool 后另算。

## 固定配置

- Queries: `50`
- Vector/BM25/Fusion/Rerank/Final top-k: `50/50/50/50/10`
- RRF k: `60`
- Weighted vector/BM25: `0.65/0.35`
- Reranker batch/max length: `8/512`
- Reranker policy: strict; model unavailable or scoring failure aborts the run
- BM25 scope: part000（64,853 chunks）；305 万 chunks 的全量分片 BM25 延后到 T015
- Query execution: query understanding、vector search 和 BM25 search 每条只执行一次，三种融合复用相同两路候选
- Final evidence: 按 document key 去重，每篇文档保留最高分 chunk

## 运行与完整性

- Fusion candidates: `7500`
- Final evidence rows: `1500`
- Annotation pool query-document rows: `575`
- `150` 个 query-strategy 分组全部使用真实 reranker；无 fallback、无缺失 reranker score；每组最终 `10` 篇文档且 document key 唯一。

| Strategy | Query groups | Final rows | Shared retrieval s/query | Fusion s/query | Rerank+score s/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| simple | 50 | 500 | 4.154 | 0.0009 | 0.238 |
| rrf | 50 | 500 | 4.154 | 0.0002 | 0.235 |
| weighted | 50 | 500 | 4.154 | 0.0002 | 0.235 |

## 候选与最终结果重合度

| Stage | Pair | Mean overlap rate | Mean Jaccard |
| --- | --- | ---: | ---: |
| pre_doc_top50 | simple vs rrf | 0.9970 | 0.9864 |
| pre_doc_top50 | simple vs weighted | 0.9255 | 0.7914 |
| pre_doc_top50 | rrf vs weighted | 0.9271 | 0.7946 |
| pre_chunk_top50 | simple vs rrf | 0.9896 | 0.9798 |
| pre_chunk_top50 | simple vs weighted | 0.8756 | 0.7826 |
| pre_chunk_top50 | rrf vs weighted | 0.8768 | 0.7847 |
| final_doc_top5 | simple vs rrf | 0.9600 | 0.9333 |
| final_doc_top5 | simple vs weighted | 0.8320 | 0.7383 |
| final_doc_top5 | rrf vs weighted | 0.8360 | 0.7450 |
| final_doc_top10 | simple vs rrf | 0.9780 | 0.9612 |
| final_doc_top10 | simple vs weighted | 0.8600 | 0.7645 |
| final_doc_top10 | rrf vs weighted | 0.8600 | 0.7645 |

实测中 simple 与 RRF 的最终 Top-10 文档集合在全部查询上完全一致：`False`。weighted 与另外两种策略的最终 Top-10 平均 overlap rate 为约 `0.888`，说明固定 reranker 能消除 simple/RRF 的排序差异，但 weighted 的候选集合差异仍会传递到最终结果。该观察只说明候选敏感性，不等于相关性质量优劣。

## 双语配对一致性

| Strategy | EN-ZH pairs | Mean Top-10 doc overlap | Mean Jaccard |
| --- | ---: | ---: | ---: |
| simple | 6 | 0.2500 | 0.1539 |
| rrf | 6 | 0.2500 | 0.1539 |
| weighted | 6 | 0.2000 | 0.1180 |

双语一致性仍偏低，说明当前中文查询增强覆盖有限；Q043 已采用中英术语并列以避免英文语料上的 BM25 空结果。这是后续 query rewrite/跨语言检索的改进点，不在本轮扩展范围内。

## 元数据过滤解释

`filter_applied=true` 仅表示 query understanding 生成了可直接执行的 `where_filter` 并传给两路检索。主 benchmark 中 Q050 执行 `article_type=research-article` 硬过滤；年份范围和 section 条件只记录在 `filter_plan`，不能宣称已执行。

预检显示 Q045 的 `pub_year=2003 AND article_type=research-article` 在全量 Chroma 上可返回 50 条、part000 BM25 返回 25 条，但 Top-50 被少数文档的多个 chunks 占据，文档去重后不足 10；Q046 的 `PLoS ONE` 在 part000 BM25 语料中不存在。因此 Q045/Q046 在主融合公平比较中只审计 filter plan，不执行硬过滤，避免把语料范围差异或 chunk 多样性问题误写成融合策略效果。

## 人工标注下一步

在 `fusion_with_reranker_annotation_pool_full_bm25_top50_q50.csv` 的 `relevance_label` 列填 0/1/2。完成后才能计算各策略 Recall@10、MRR@10、nDCG@10，并做有依据的质量选择。
