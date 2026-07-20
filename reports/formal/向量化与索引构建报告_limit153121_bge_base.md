# 向量化与索引构建报告（limit153121_bge_base）

## 1. 任务目标

本轮读取既有全文 chunk Parquet，不重新解析 XML、不重新切 chunk，使用 BAAI/bge-base-en-v1.5 生成归一化向量，并写入持久化 Chroma collection。

## 2. 输入 chunk 数据集

- Manifest: `/root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_manifest.csv`
- Expected chunks: `3057078`
- Selected parts: `52`
- Max chunks: `none`

## 3. 模型与向量配置

- Embedding model: `BAAI/bge-base-en-v1.5`
- Embedding dimension: `768`
- Device: `cuda`
- normalize_embeddings: `true`
- Chroma metric: `cosine`
- 文档 embedding 不加 query instruction；查询脚本会自动添加 BGE query instruction。

## 4. Chroma Collection

- Collection: `pmc_fulltext_bge_base_limit153121`
- Persist dir: `/root/autodl-tmp/medical_rag/artifacts/indexes/chroma/pmc_fulltext_bge_base_limit153121`
- Collection metadata: `{"hnsw:space": "cosine"}`

## 5. 写入规模与校验

- Total vectors indexed: `3057078`
- Collection count: `3057078`
- Count matched expected: `True`
- Elapsed: `6h 33m`
- Index size: `49345.952 MB` / `48.189 GB`

## 6. Chunk Token 统计

- mean: `333.877`
- min: `16`
- max: `512`
- p95: `416.0`

## 7. Metadata 字段

`doc_id, chunk_index, total_chunks, source_title, journal, pub_date, pub_year, pmid, pmcid, article_type, section_title, section_title_norm, split_strategy, quality_decision, source_file, token_count`

## 8. Part 摘要

| part_id | part_file | chunk_count | status | indexed_at |
| --- | --- | --- | --- | --- |
| 0 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part000.parquet | 64853 | complete | 2026-07-08T01:15:27 |
| 1 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part001.parquet | 65102 | complete | 2026-07-08T01:23:02 |
| 2 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part002.parquet | 57604 | complete | 2026-07-08T01:29:49 |
| 3 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part003.parquet | 65697 | complete | 2026-07-08T01:37:45 |
| 4 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part004.parquet | 67365 | complete | 2026-07-08T01:45:49 |
| 5 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part005.parquet | 70743 | complete | 2026-07-08T01:54:13 |
| 6 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part006.parquet | 69791 | complete | 2026-07-08T02:02:46 |
| 7 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part007.parquet | 71358 | complete | 2026-07-08T02:11:19 |
| 8 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part008.parquet | 80908 | complete | 2026-07-08T02:21:02 |
| 9 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part009.parquet | 13248 | complete | 2026-07-08T02:22:49 |
| 10 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part010.parquet | 15618 | complete | 2026-07-08T02:24:41 |
| 11 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part011.parquet | 7292 | complete | 2026-07-08T02:25:33 |

## 9. 结论

- chroma_created: `true`
- embedding_created: `true`
- 后续验证脚本: `scripts/13_validate_chroma_index.py`
- 手动查询脚本: `scripts/14_query_chroma_index.py`
