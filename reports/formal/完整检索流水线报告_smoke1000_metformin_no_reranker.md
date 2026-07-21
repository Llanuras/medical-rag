# 完整检索流水线报告（smoke1000_metformin_no_reranker）

## 查询与状态

- Query: `二甲双胍对心血管疾病有何影响？`
- Status: `ok`
- Vector candidates: `5`
- Keyword candidates: `5`
- Fused candidates: `10`
- Reranker requested: `False`
- Reranker used: `False`

## 融合策略说明

- `simple`：按向量结果、BM25 结果的输入顺序合并并按 `chunk_id` 去重，仅作基线。
- `rrf`：使用倒数排名融合，不直接混合 Chroma 与 BM25 的异构原始分数，是当前默认推荐。
- `weighted`：对单次查询的两路分数分别 Min-Max 归一化后加权；向量/BM25 权重 `0.65/0.35` 只是启发式初始值，不是固定标准。

## 多准则排序说明

- 相关性是主信号；reranker 禁用或不可用时，使用归一化 fusion score 降级。
- authority score 是可审计的 journal prior，不是影响因子。
- recency score 只做软排序，不做硬过滤；当前 deprecated 语料年份整体偏旧，不应把该分数解读为文献质量结论。

## Evidence Top-5

| Rank | Title | Journal | Year | Sources | Final score |
| --- | --- | --- | --- | --- | --- |
| 1 | Candidate Gene Association Study in Type 2 Diabetes Indicates a Role for Genes Involved in β-Cell Function as Well as Insulin Action | PLoS Biology | 2003 | ['keyword'] | 0.7350 |
| 2 | A cardiologic approach to non-insulin antidiabetic pharmacotherapy in patients with heart disease | Cardiovascular Diabetology | 2009 | ['vector'] | 0.7125 |
| 3 | Structural Mechanism Shows How Transferrin Receptor Binds Multiple Ligands and Sheds Light on a Hereditary Iron Disease | PLoS Biology | 2003 | ['keyword'] | 0.5777 |
| 4 | A cardiologic approach to non-insulin antidiabetic pharmacotherapy in patients with heart disease | Cardiovascular Diabetology | 2009 | ['vector'] | 0.5552 |
| 5 | Candidate Gene Association Study in Type 2 Diabetes Indicates a Role for Genes Involved in β-Cell Function as Well as Insulin Action | PLoS Biology | 2003 | ['keyword'] | 0.4255 |

## Warnings

- reranker disabled; normalized fusion_score is used as relevance_score

## 范围边界

当前阶段只输出 retrieval evidence list，不生成医学答案。
