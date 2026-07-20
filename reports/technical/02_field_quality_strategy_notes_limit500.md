# 字段完整性、清洗策略与关键字段分析说明

## 1. 本阶段分析目标

RAG 构建前需要确认字段完整性、文本质量和 metadata 可用性。字段缺失会影响向量库文本构造，metadata 缺失会影响后续过滤和溯源，低质量文本会降低检索和回答质量。

## 2. 分析维度

本阶段分析了字段缺失率、abstract 可用性、基础文本质量、metadata 可用性和原文溯源能力。

## 3. 字段缺失的可选处理策略

- 直接丢弃缺失样本：优点是保证入库文本质量；缺点是可能损失样本量。
- 使用 body 替代 abstract：优点是保留文献信息；缺点是 body 较长且可能需要额外切分。
- 使用 title 作为弱文本：优点是保留主题信息；缺点是信息量不足，不适合作为核心向量文本。
- 保留为 metadata 但不入库：优点是保留追溯信息；缺点是不能参与语义检索。
- 人工补全或二次抓取：优点是质量高；缺点是成本较高，不适合当前批量准备阶段。

## 4. abstract 缺失处理策略

当前 500 篇中 abstract 缺失率为 `22.2000%`。因此建议：分层处理：body 替代、title-only metadata、无文本丢弃。

## 5. 基础质量清洗策略

- 极短 abstract：标记为 `keep_with_warning`，后续入库时可保留但需人工抽查。
- 乱码文本：标记为 `need_review`，避免污染向量库。
- 重复 title+abstract：标记为 `need_review`，避免重复文档影响检索排序。
- 空正文：如果 title+abstract 可用，可用于摘要级 RAG；若核心文本为空则不入库。

## 6. 关键 metadata 字段策略

- title 缺失率：`0.0000%`，可用于增强检索文本。
- journal 可用率：`100.0000%`，可作为期刊过滤器；如果可用率不足，未来实现“检索近5年某一期刊上的文献”需要 fallback。
- pub_year 可用率：`100.0000%`，可作为时间过滤器。
- pmid 可用率：`72.6000%`，可用于 PubMed 追溯。
- pmcid 可用率：`100.0000%`，可用于 PMC 追溯。

## 7. 当前 500 篇数据的实际结论

详见 `artifacts/metrics/t002_corpus_analysis/missing_rate_limit500.csv`、`artifacts/metrics/t002_corpus_analysis/quality_summary_limit500.csv` 和 `artifacts/metrics/t002_corpus_analysis/metadata_summary_limit500.csv`。所有数值均来自真实解析的 500 篇 XML。

## 8. 对后续 RAG 构建的影响

建议入库样本使用 `quality_decision` 为 `keep` 或 `keep_with_warning` 且 `text_title_abstract` 非空的记录。`journal`、`pub_year`、`pmid`、`pmcid`、`source_file` 应作为 metadata 保存，用于后续过滤、citation 和 source tracking。
