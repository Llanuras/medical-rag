# PMC000 全文 chunk 小样本测试详细报告（test3028_pmc000）

## 1. 测试目标与边界

本次测试用于验证 `scripts/11_chunk_oa_comm_153121.py` 在正式全量运行前的可用性。测试对象为 `data/raw/pmc_oa_comm/PMC000xxxxxx` 切片完整 `3028` 篇 XML。

本次只验证全文级 chunk 数据集生成，不做摘要级 chunk，不做 embedding，不做 Chroma 入库，不做 RAG 问答，也不生成 PDF。正式 `limit153121` 全量尚未运行。

## 2. 输入、环境与命令

| 项目 | 内容 |
|---|---|
| 输入目录 | `data/raw/pmc_oa_comm/PMC000xxxxxx` |
| 输入 XML 数 | 3028 |
| 输出前缀 | `test3028_pmc000` |
| tokenizer | `sentence-transformers/all-MiniLM-L6-v2` |
| tokenizer 缓存 | `artifacts/models/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| chunk_size | 400 |
| chunk_overlap | 80 |
| whole_doc_token_limit | 512 |
| batch_size | 500 |

执行命令：

```bash
python scripts/11_chunk_oa_comm_153121.py \
  --data_dir data/raw/pmc_oa_comm/PMC000xxxxxx \
  --output_prefix test3028_pmc000 \
  --batch_size 500 \
  --chunk_size 400 \
  --chunk_overlap 80 \
  --whole_doc_token_limit 512 \
  --force
```

语法检查已通过：

```bash
python -m py_compile scripts/11_chunk_oa_comm_153121.py
```

## 3. 输出产物

本次共生成 7 个 Parquet part，manifest 和各类统计、预览、质量报告均已生成。

| 类型 | 路径 |
|---|---|
| Parquet shards | `artifacts/datasets/chunks/pmc_chunks_test3028_pmc000_part000.parquet` 至 `part006.parquet` |
| manifest | `artifacts/datasets/chunks/pmc_chunks_test3028_pmc000_manifest.csv` |
| 总览统计 | `artifacts/metrics/t007_chunking/chunk_summary_test3028_pmc000.csv` |
| 路由统计 | `artifacts/metrics/t007_chunking/chunk_route_summary_test3028_pmc000.csv` |
| token 统计 | `artifacts/metrics/t007_chunking/chunk_token_length_stats_test3028_pmc000.csv` |
| section title Top80 | `artifacts/metrics/t007_chunking/chunk_section_title_top80_test3028_pmc000.csv` |
| 质量 flags | `artifacts/metrics/t007_chunking/chunk_quality_flags_test3028_pmc000.csv` |
| excluded 表 | `artifacts/metrics/t007_chunking/excluded_documents_test3028_pmc000.csv` |
| 预览 Markdown | `reports/samples/chunk_preview_test3028_pmc000.md` |
| 预览 JSONL | `reports/samples/chunk_preview_test3028_pmc000_sample.jsonl` |
| 质量验证报告 | `reports/formal/文档解析与分割质量验证_test3028_pmc000.md` |
| 日志 | `logs/11_chunk_oa_comm_test3028_pmc000.log` |

Parquet 总大小约 `39.96 MB`。`part000` 已通过 pandas 读取验证。

## 4. 核心统计结果

| 指标 | 数值 |
|---|---:|
| 输入 XML | 3028 |
| processed_documents | 3028 |
| chunked_documents | 3028 |
| excluded_documents | 0 |
| total_chunks | 66340 |
| chunks/doc mean | 21.91 |
| chunks/doc median | 20 |
| chunks/doc p75 | 29 |
| chunks/doc p90 | 39 |
| chunks/doc p95 | 46 |
| chunks/doc p99 | 63 |
| chunks/doc max | 103 |
| chunk token mean | 326.20 |
| chunk token median | 377 |
| chunk token p75 | 395 |
| chunk token p90 | 406 |
| chunk token p95 | 413 |
| chunk token p99 | 425 |
| chunk token max | 520 |

整体分布说明：绝大多数 chunk 控制在较稳定的 300-425 tokens 区间，p95 为 413，说明 `chunk_size=400` 与 `chunk_overlap=80` 对 PMC000 切片基本有效。

## 5. 分割路由结果

| split_strategy | 文献数 | 文献占比 | chunk 数 | chunk 占比 |
|---|---:|---:|---:|---:|
| whole_document_under_512 | 30 | 0.99% | 30 | 0.05% |
| semantic_section | 2694 | 88.97% | 64894 | 97.82% |
| recursive_fallback_no_section | 304 | 10.04% | 1416 | 2.13% |

PMC000 切片中，`semantic_section` 是绝对主路径，说明该切片大多数文献具有可利用的 XML section title。`recursive_fallback_no_section` 处理 304 篇，所有这些 fallback 文献均为多 chunk 文献，说明 overlap 兜底路径实际被覆盖。

与历史 T005 的 3028 Chroma 路由 `28 / 2695 / 305` 相比，本次结果为 `30 / 2694 / 304`。差异很小，主要来自本次按 M3 要求重新从原始 XML 解析 `title + body` 并采用新的全文级 dataset 字段口径，不能简单视为异常。

## 6. Parquet 分片情况

| part | 文献数 | chunk 数 | 大小 MB | 起止 doc_index |
|---|---:|---:|---:|---|
| part000 | 500 | 10028 | 6.168 | 0-499 |
| part001 | 500 | 11554 | 6.849 | 500-999 |
| part002 | 500 | 10400 | 6.323 | 1000-1499 |
| part003 | 500 | 10684 | 6.447 | 1500-1999 |
| part004 | 500 | 11447 | 6.972 | 2000-2499 |
| part005 | 500 | 11544 | 6.779 | 2500-2999 |
| part006 | 28 | 683 | 0.423 | 3000-3027 |

分片写出符合预期：每 500 篇一个 part，最后一个 part 只包含剩余 28 篇。manifest 中记录了每个 part 的起止 doc_id 和文件大小。

## 7. Chunk 质量验证

| 检查项 | 结果 |
|---|---:|
| chunk_id 全局唯一 | True |
| duplicate_chunk_ids | 0 |
| doc_id_missing | 0 |
| source_title_missing | 0 |
| pmcid_missing | 0 |
| source_file_missing | 0 |
| chunk_index 非连续文献数 | 0 |
| total_chunks 不匹配文献数 | 0 |
| max_index mismatch 文献数 | 0 |
| token_count <= 0 | 0 |
| 空 text chunk | 0 |
| 乱码 chunk | 0 |
| semantic_section 缺失 section_title | 0 |
| token_count > 512 | 3 |
| token_count < 20 | 11 |

结论：ID、索引、追溯 metadata、空文本、乱码和 section title 保留方面均通过。唯一需要在正式全量前处理的是 3 个超过 512 tokens 的 chunk。

## 8. 超过 512 tokens 的 chunk

本次共有 3 个 chunk 超过 512 tokens，均来自 `whole_document_under_512` 路由：

| chunk_id | token_count | source_title |
|---|---:|---|
| `PMCID:PMC535702::chunk_00000` | 520 | Sleep Duration Affects Appetite-Regulating Hormones |
| `PMCID:PMC545211::chunk_00000` | 516 | Meningitis and Climate in West Africa |
| `PMCID:PMC545215::chunk_00000` | 518 | Three More Learning Points |

原因判断：路由条件使用 `title + body <= 512`，但最终 chunk text 会额外加入：

```text
Title: ...
Section: ...
Text:
...
```

这部分上下文包装会额外消耗 tokens，导致少数接近阈值的 whole-document chunk 超过 512。比例仅为 `0.0045%`，但如果下一阶段直接使用 `all-MiniLM-L6-v2` embedding，建议正式全量前修正为“以最终 chunk text token_count <= 512 为验收约束”。

## 9. 极短 chunk

本次 `token_count < 20` 的 chunk 共 11 个，主要分两类：

1. 只有标题或极短正文的短文，如 `Neuroscience Networks`、`Niche Markets`、`Talking Science`。
2. 论文尾部声明类 section，如 `Competing interests: None declared`、`Authors' contributions: Single author`。

这些 chunk 不是空文本，也不是乱码。是否保留取决于后续检索目标：如果希望保留论文所有可追溯结构，可保留；如果 embedding 成本敏感，正式全量前可考虑对声明类短 section 加 `quality_decision=keep_with_warning` 或后续检索时降权。

## 10. Section Title 分布

Top section titles 体现出医学/生物医学论文结构被较好保留：

| section_title | chunk_count |
|---|---:|
| Results | 17040 |
| Discussion | 9790 |
| Methods | 8665 |
| Background | 5832 |
| Results and discussion | 2340 |
| Introduction | 2198 |
| Materials and Methods | 1821 |
| Conclusions | 1713 |
| Authors' contributions | 1563 |
| recursive_fallback | 1416 |

这说明 `semantic_section` 路由不仅命中率高，而且保留了后续检索可用的结构化 metadata。`recursive_fallback` 作为无章节兜底也被单独标记，便于后续质量抽查。

## 11. 预览覆盖情况

`reports/samples/chunk_preview_test3028_pmc000.md` 已覆盖：

- `whole_document_under_512` 样本；
- `semantic_section` 样本；
- `recursive_fallback_no_section` 样本；
- 多 chunk 文献连续 chunk 示例；
- chunk metadata 示例；
- 同一 doc_id 下连续多个 chunk 示例。

未覆盖：

- title 缺失但保留样本；
- `excluded no_body` 样本。

未覆盖原因不是脚本跳过，而是 PMC000 切片中未命中这些情况。本项目全量 153121 篇中已知存在 title/body 缺失样本，因此正式全量前最好补一个针对边界文献的小型验证，确认 excluded 表和 title fallback 在真实缺失样本上也按预期工作。

## 12. 处理耗时与断点续跑

首次 `--force` 运行已完成 7 个 Parquet part 写出。日志显示第 7 个 part 写出时累计约 615 秒。随后因环境缺少 `tabulate`，在 Markdown 报告渲染阶段失败；脚本已改为内置 Markdown 表格渲染，不再依赖 `tabulate`。

非 `--force` 续跑时，脚本跳过 7 个已存在 part，只重建统计、预览和报告；续跑成功，`chunk_summary` 中记录该阶段耗时约 71 秒。

这说明断点续跑机制基本可用：已完成 part 可保留并跳过，失败后可以继续补齐下游统计产物。

## 13. 风险与修正建议

### 必修正

正式全量前建议修正 final chunk token 上限：

- 以最终 `text` 的 `token_count <= 512` 作为硬约束；
- 或将 whole-document 路由阈值从 512 下调到 480；
- 更稳妥方案是：生成 chunk text 后若 token_count > 512，则对正文部分进入 recursive split，并保留 Title/Section 包装。

这能消除本次发现的 3 个 whole-document 超限样本。

### 建议补测

在正式 153121 全量前，建议从全量轻量表或原始 XML 中抽取少量真实边界文献：

- title 缺失但 body 存在；
- body 缺失但 title/abstract 存在；
- title、abstract、body 均缺失；
- 如有重复 PMCID 或异常 XML，也纳入边界测试。

PMC000 切片没有命中这些边界，因此当前只能证明主路径可靠，不能完全证明缺失字段分支在真实样本中已覆盖。

## 14. 是否可以开启正式全量

当前小样本测试结论是：主流程可以跑，Parquet 分片、manifest、metadata、ID、索引连续性和断点续跑都通过；但不建议“原样”直接开启正式 153121 全量。

建议先做一个小修正，让最终 chunk text 的 token_count 不超过 512，并补跑 PMC000 3028 篇复测。如果复测达到：

- `token_count > 512 = 0`；
- `chunk_id` 唯一；
- `chunk_index` 连续；
- pandas 可读；
- route/manifest/report 正常生成；

则可以开启正式 `limit153121` 全量测试。

## 15. 结论

本次 `test3028_pmc000` 是一次有效的小样本验收。它证明了全文 chunk 生成脚本的主干设计成立：能够从原始 XML 解析正文，按三路由切分，生成可追溯的 Parquet shards，并产出统计、预览和质量验证结果。

正式全量运行前的唯一硬性改进点是消除最终 chunk text 超过 512 tokens 的边界问题。完成该修正并复测后，正式 153121 全量运行风险较低。
