# 全文 Chunk 处理日志与统计报告（limit153121）

## 1. 交付物完整性结论

本轮“文档解析与分割工作”的核心交付物已经齐全：

| 交付要求 | 当前状态 | 对应文件 |
|---|---|---|
| 文本块数据集 | 已生成 | `artifacts/datasets/chunks/pmc_chunks_limit153121_part000.parquet` 至 `part051.parquet` |
| 数据集 manifest | 已生成 | `artifacts/datasets/chunks/pmc_chunks_limit153121_manifest.csv` |
| JSONL 预览样例 | 已生成 | `reports/samples/chunk_preview_limit153121_sample.jsonl` |
| 处理日志 | 已生成 | `logs/11_chunk_oa_comm_limit153121.log` |
| 统计表 | 已生成 | `artifacts/metrics/t007_chunking/chunk_summary_limit153121.csv` 等 |
| excluded 文献表 | 已生成 | `artifacts/metrics/t007_chunking/excluded_documents_limit153121.csv` |
| 质量验证报告 | 已生成 | `reports/formal/文档解析与分割质量验证_limit153121.md` |
| Markdown 预览 | 已生成 | `reports/samples/chunk_preview_limit153121.md` |

## 2. 运行配置

| 配置项 | 数值 |
|---|---|
| data_split | `limit153121` |
| 输入目录 | `data/raw/pmc_oa_comm` |
| 输入 XML 数 | 153121 |
| tokenizer | `sentence-transformers/all-MiniLM-L6-v2` |
| tokenizer 缓存 | `artifacts/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| chunk_size | 400 |
| chunk_overlap | 80 |
| whole_doc_token_limit | 512 |
| batch_size | 3000 |
| chroma_created | false |
| embedding_created | false |

## 3. 处理日志摘要

原始日志文件为：

```text
logs/11_chunk_oa_comm_limit153121.log
```

日志显示全量处理正常结束，最后状态为 `DONE`。关键日志结果如下：

| 指标 | 数值 |
|---|---:|
| Input XML | 153121 |
| Chunked documents | 133539 |
| Excluded documents | 19582 |
| Total chunks | 3057078 |
| Chunk token p95 | 416 |
| Token >512 chunks | 0 |
| Duplicate chunk IDs | 0 |
| Manifest | `artifacts/datasets/chunks/pmc_chunks_limit153121_manifest.csv` |

脚本运行阶段按 part 和 overall 同时输出进度、速度和 ETA。最后一个 part 为 `part051`，处理 doc index `153000-153120`，说明 153121 篇 XML 已全部覆盖。

日志中的 Arrow `sysctlbyname failed` 信息来自 macOS sandbox 下的 CPU 信息探测警告，不影响 Parquet 读写和结果完整性。

## 4. 数据集输出结构

正式 chunk 数据集以 52 个 Parquet shard 保存：

```text
artifacts/datasets/chunks/pmc_chunks_limit153121_part000.parquet
...
artifacts/datasets/chunks/pmc_chunks_limit153121_part051.parquet
```

Manifest 路径：

```text
artifacts/datasets/chunks/pmc_chunks_limit153121_manifest.csv
```

Manifest 覆盖情况：

| 指标 | 数值 |
|---|---:|
| part 数 | 52 |
| manifest 文献数合计 | 153121 |
| manifest chunk 数合计 | 3057078 |
| Parquet 总大小 | 1801.509 MB |
| Parquet 总大小 | 1.759 GB |
| 每 part chunk 数均值 | 58789.96 |
| 每 part chunk 数中位数 | 68362 |
| 每 part chunk 数最小值 | 2942 |
| 每 part chunk 数最大值 | 80908 |

最大 chunk part：

| part_id | document_count | chunk_count | file_size_mb |
|---:|---:|---:|---:|
| 8 | 3000 | 80908 | 47.440 |
| 27 | 3000 | 79874 | 48.117 |
| 32 | 3000 | 79142 | 46.594 |
| 28 | 3000 | 78669 | 46.601 |
| 29 | 3000 | 77276 | 46.100 |

最小 chunk part：

| part_id | document_count | chunk_count | file_size_mb | 说明 |
|---:|---:|---:|---:|---|
| 51 | 121 | 2942 | 1.747 | 最后剩余 part |
| 11 | 3000 | 7292 | 4.315 | 该批可用正文或长文比例较低 |
| 46 | 3000 | 11845 | 4.215 | 该批 chunk 数较少 |
| 9 | 3000 | 13248 | 7.914 | 该批 chunk 数较少 |
| 10 | 3000 | 15618 | 9.127 | 该批 chunk 数较少 |

## 5. Chunk 字段结构

每条 chunk 至少包含以下字段：

| 字段 | 作用 |
|---|---|
| `chunk_id` | 全局唯一 chunk ID，格式为 `{doc_id}::chunk_{chunk_index:05d}` |
| `text` | 最终用于 embedding/retrieval 的文本，包含 Title、Section、Text |
| `doc_id` | 文献主 ID，优先使用 PMCID |
| `chunk_index` | 文档内 chunk 序号，从 0 开始 |
| `total_chunks` | 当前文档实际生成 chunk 总数 |
| `source_title` | 原始标题 metadata |
| `token_count` | 最终 chunk text 的 tokenizer token 数 |
| `source_file` | 原始 XML 文件路径 |
| `journal` | 期刊 metadata |
| `pub_date` / `pub_year` | 出版时间 metadata |
| `pmid` | PubMed 追溯 ID |
| `pmcid` | PMC 追溯 ID |
| `article_type` | 文章类型 |
| `section_title` | XML section title 或 fallback section |
| `section_title_norm` | section_title 标准化文本 |
| `split_strategy` | 分割路由 |
| `quality_decision` | 质量标记 |
| `title_missing` | 标题是否缺失 |
| `body_missing` | 正文是否缺失 |
| `chunk_char_len` | chunk 字符长度 |
| `section_index` | section 序号 |
| `section_chunk_index` | section 内 chunk 序号 |

最终 chunk text 格式为：

```text
Title: {title_or_fallback}
Section: {section_title_or_strategy}
Text:
{chunk_body_text}
```

## 6. 总体处理规模

| 指标 | 数值 |
|---|---:|
| original_xml | 153121 |
| processed_documents | 153121 |
| chunked_documents | 133539 |
| excluded_documents | 19582 |
| total_chunks | 3057078 |
| token_count_over_512 | 0 |
| token_count_le_zero | 0 |
| empty_text_chunks | 0 |
| duplicate_chunk_ids | 0 |
| doc_id_missing | 0 |
| source_title_missing | 11 |
| pmcid_missing | 0 |
| source_file_missing | 0 |
| non_continuous_doc_count | 0 |
| total_chunks_mismatch_doc_count | 0 |
| max_index_mismatch_doc_count | 0 |

整体结论：所有输入 XML 均完成处理；有正文的文献进入全文 chunk dataset；无正文或无文本文献进入 excluded 表。chunk ID、doc ID、PMCID/source_file 追溯、chunk_index 连续性和 total_chunks 一致性均通过质量验证。

## 7. Excluded 文献统计

| excluded_reason | 文献数 | 说明 |
|---|---:|---|
| `no_body_for_fulltext` | 19567 | 有 title/abstract 等文本但无正文；本轮只做全文级 chunk，因此不生成 chunk |
| `drop_no_text` | 15 | title、abstract、body 均无可用文本 |
| 合计 | 19582 | 全部已记录到 excluded 表 |

excluded 表路径：

```text
artifacts/metrics/t007_chunking/excluded_documents_limit153121.csv
```

字段包括 doc_id、pmid、pmcid、title、source_file、excluded_reason、has_title、has_abstract、has_body、title_token_len、abstract_token_len、body_token_len、fallback_available、note。

## 8. 分割路由统计

| split_strategy | 文献数 | chunk 数 | 文献占比 | chunk 占比 |
|---|---:|---:|---:|---:|
| `semantic_section` | 124996 | 3027781 | 81.63% | 99.04% |
| `recursive_fallback_no_section` | 5699 | 26156 | 3.72% | 0.86% |
| `whole_document_under_512` | 2844 | 3141 | 1.86% | 0.10% |
| `no_body_for_fulltext` | 19567 | 0 | 12.78% | 0.00% |
| `drop_no_text` | 15 | 0 | 0.01% | 0.00% |

解释：

- `semantic_section` 是主路径，说明大多数有正文文献具备可用 XML section title。
- `recursive_fallback_no_section` 覆盖无明确 section title 的正文文献。
- `whole_document_under_512` 覆盖少量短正文文献；其中部分短文在最终 Title/Section/Text 包装后仍被拆为多个 chunk，因此文献数 2844、chunk 数 3141。
- `no_body_for_fulltext` 与 `drop_no_text` 只进入 excluded 表，不生成 chunk。

## 9. 文本块大小统计

### 9.1 chunks/doc 分布

| 指标 | 数值 |
|---|---:|
| count | 133539 |
| mean | 22.89 |
| median | 22 |
| p75 | 32 |
| p90 | 42 |
| p95 | 49 |
| p99 | 67 |
| max | 741 |

平均每篇有正文文献生成约 22.9 个 chunks。p95 为 49，说明绝大多数文献的 chunk 数处于可控范围。最大值 741 来自超长文献 `PMCID:PMC2848993`，标题为 `Nomenclature for factors of the HLA system, 2010`，属于长表/命名法类内容，不是索引错误。

### 9.2 chunk token_count 分布

| 指标 | 数值 |
|---|---:|
| count | 3057078 |
| mean | 333.88 |
| median | 379 |
| p75 | 397 |
| p90 | 409 |
| p95 | 416 |
| p99 | 434 |
| max | 512 |

本轮最终以完整 chunk text 计算 token_count，即包含 `Title`、`Section` 和正文片段。所有 chunk 均满足 `token_count <= 512`，可作为下一阶段 `all-MiniLM-L6-v2` embedding 输入。

## 10. Section Title 分布

Top section title 如下：

| section_title | chunk_count |
|---|---:|
| Results | 853611 |
| Discussion | 426437 |
| Methods | 332761 |
| Background | 164022 |
| Introduction | 132707 |
| Materials and Methods | 132020 |
| Results and Discussion | 77467 |
| Results and discussion | 67210 |
| Authors' contributions | 57240 |
| Materials and methods | 51395 |
| Conclusion | 48363 |
| Competing interests | 42996 |
| Supporting Information | 30699 |
| Supplementary Material | 29329 |
| recursive_fallback | 26156 |

结论：主要 chunk 来自 Results、Discussion、Methods、Background、Introduction 等论文核心结构，语义章节保留效果良好。`recursive_fallback` 单独计数，便于后续抽查无 section title 文献的切分质量。

## 11. 质量验证结果

| 检查项 | 结果 |
|---|---:|
| chunk_id_global_unique | True |
| doc_id_missing_count | 0 |
| chunk_index_non_continuous_doc_count | 0 |
| total_chunks_mismatch_doc_count | 0 |
| token_count_over_512_count | 0 |
| empty_text_chunk_count | 0 |
| short_chunk_count_token_lt_20 | 69 |
| garbled_chunk_count | 10 |
| source_title_missing_count | 11 |
| pmcid_missing_count | 0 |
| source_file_missing_count | 0 |
| semantic_section_without_section_title | 0 |
| recursive_fallback_configured_overlap | 80 |
| recursive_fallback_multi_chunk_doc_count | 5699 |

质量结论：

- 核心结构一致性全部通过：chunk ID 唯一、doc ID 不缺失、chunk_index 连续、total_chunks 正确。
- 所有 chunk 都满足 embedding token 上限：`token_count > 512 = 0`。
- 追溯字段完整：pmcid 和 source_file 均不缺失。
- 语义章节元数据完整：semantic_section chunk 均保留 section_title。

需要后续决策的少量质量标记：

| 问题 | 数量 | 建议 |
|---|---:|---|
| source_title_missing | 11 | 已使用 `[Missing title: PMCID:...]` fallback，可保留 |
| garbled_chunk | 10 | embedding 前建议抽查，必要时过滤 |
| token_count < 20 | 69 | 多为极短文献或声明类 section，可保留或后续降权 |

## 12. Quality Decision 分布

| quality_decision | chunk 数 |
|---|---:|
| keep | 3014465 |
| keep_with_warning | 42504 |
| need_review | 109 |

绝大多数 chunk 标记为 `keep`。`keep_with_warning` 主要对应标题缺失、摘要缺失、正文较短或轻微质量风险。`need_review` 数量很少，可在进入 embedding 前重点抽查。

## 13. 预览与抽样检查

预览文件：

```text
reports/samples/chunk_preview_limit153121.md
reports/samples/chunk_preview_limit153121_sample.jsonl
```

Markdown 预览包含：

- `whole_document_under_512` 样本；
- `semantic_section` 样本；
- `recursive_fallback_no_section` 样本；
- 多 chunk 文献样本；
- title 缺失但保留的样本；
- chunk metadata 示例；
- 同一 doc_id 下连续多个 chunk 的示例。

## 14. 总结

`limit153121` 全文 chunk 数据集已经完成，且满足正式进入下一阶段 embedding 的主要技术条件：

- 输入覆盖完整；
- 有正文文献全部生成 chunk；
- 无正文文献全部进入 excluded 表；
- chunk text 均包含 Title/Section/Text 上下文；
- token_count 全部不超过 512；
- chunk_id 全局唯一；
- doc_id、PMCID、source_file 可追溯；
- chunk_index 与 total_chunks 一致；
- 统计、日志、预览和质量验证文件齐全。

本轮正式产物可以作为后续医学 RAG 向量化输入。
