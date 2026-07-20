# 最小 embedding/Chroma 可用性测试说明

## 1. 本阶段目标

本阶段只验证 embedding + Chroma pipeline 是否可行，不做正式大规模建库。

## 2. 为什么需要 embedding 测试

后续 RAG 需要将文本转为向量，并写入向量数据库，才能执行 similarity_search。因此需要先用少量样本确认链路可用。

## 3. 为什么只用 5-10 条样本

本次任务重点是数据分析和策略设计，不是大规模向量化。选择 10 条样本可以验证模型下载、向量生成、Chroma 写入和检索，同时控制耗时和复杂度。

## 4. 可选策略

- 不做 embedding 测试：速度最快，但无法确认向量库链路可用。
- 最小 embedding 测试：能验证关键链路，成本低，是当前选择。
- 对 500 篇全部 embedding：可生成测试库，但应在最小测试通过后执行。
- 对全文全部 embedding：成本更高，且需要先明确全文切分策略。

## 5. 测试结果

使用 `sentence-transformers/all-MiniLM-L6-v2` 对 `10` 条样本文本写入 Chroma，并执行 `3` 个查询。真实检索结果保存在 `artifacts/metrics/t002_corpus_analysis/min_embedding_chroma_test_results_limit500.csv`。

## 6. 后续扩展建议

正式 RAG 构建时应根据 token 长度分析决定是否切分，再批量 embedding 并写入 Chroma。
