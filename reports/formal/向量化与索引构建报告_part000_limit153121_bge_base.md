# 向量化与索引构建报告（part000_limit153121_bge_base）

## 1. 任务目标

本轮读取既有全文 chunk Parquet，不重新解析 XML、不重新切 chunk，使用 BAAI/bge-base-en-v1.5 生成归一化向量，并写入持久化 Chroma collection。

## 2. 输入 chunk 数据集

- Manifest: `/root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_manifest.csv`
- Expected chunks: `64853`
- Selected parts: `1`
- Max chunks: `none`

## 3. 模型与向量配置

- Embedding model: `BAAI/bge-base-en-v1.5`
- Embedding dimension: `768`
- Device: `cuda`
- normalize_embeddings: `true`
- Chroma metric: `cosine`
- 文档 embedding 不加 query instruction；查询脚本会自动添加 BGE query instruction。

## 4. Chroma Collection

- Collection: `pmc_fulltext_bge_base_part000_limit153121`
- Persist dir: `/root/autodl-tmp/medical_rag/artifacts/indexes/chroma/pmc_fulltext_bge_base_part000_limit153121`
- Collection metadata: `{"hnsw:space": "cosine"}`

## 5. 写入规模与校验

- Total vectors indexed: `64853`
- Collection count: `64853`
- Count matched expected: `True`
- Elapsed: `7.2s`
- Index size: `1052.711 MB` / `1.028 GB`

## 6. Chunk Token 统计

- mean: `329.438`
- min: `16`
- max: `512`
- p95: `415.0`

## 7. Metadata 字段

`doc_id, chunk_index, total_chunks, source_title, journal, pub_date, pub_year, pmid, pmcid, article_type, section_title, section_title_norm, split_strategy, quality_decision, source_file, token_count`

## 8. Part 摘要

| part_id | part_file | chunk_count | status | indexed_at |
| --- | --- | --- | --- | --- |
| 0 | /root/autodl-tmp/medical_rag/artifacts/datasets/chunks/pmc_chunks_limit153121_part000.parquet | 64853 | complete | 2026-07-08T00:59:35 |

## 9. 结论

- chroma_created: `true`
- embedding_created: `true`
- 后续验证脚本: `scripts/13_validate_chroma_index.py`
- 手动查询脚本: `scripts/14_query_chroma_index.py`
