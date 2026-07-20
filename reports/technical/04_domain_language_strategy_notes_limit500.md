# 医学文本领域内容理解说明

## 1. 本阶段分析目标

RAG 前需要理解医学文本的语言风格、信息密度和结构特点，以便设计合适的 prompt、query 改写、metadata 保留和 citation 策略。

## 2. 分层抽样策略

本阶段按 `title_abstract_token_len` 将 500 篇文献分为短、中、长三组，每组抽 5 篇，共 15 篇。短文本可能信息不足，中等文本通常适合作为摘要级检索单元，长文本可能需要切分或人工确认结构。

人工阅读样本保存在：`reports/samples/sample_short_medium_long_for_review_limit500.md`。

## 3. 医学文本结构分析

结构化摘要标记统计保存在 `artifacts/metrics/t002_corpus_analysis/structured_abstract_markers_limit500.csv`。BACKGROUND/METHODS/RESULTS/CONCLUSIONS 等标记可帮助后续按语义结构切分，尤其适合结构化摘要或全文章节处理。

## 4. 术语和缩写分析

医学文献常包含疾病名、药物名、基因名、统计术语和大写缩写。当前样本中 top 缩写包括：`DNA, RNA, IL, HIV, CI, FU, MMP, CP, HDL, AR`。缩写会影响检索召回，例如同一概念可能存在全称、缩写和同义表达，因此后续 query 改写和 prompt 中应保留术语上下文。

## 5. 高频词分析

当前 abstract 高频词 top 项包括：`genes, cells, gene, cell, expression, patients, data, protein, human, both`。高频词反映当前样本主题分布，但普通科研词汇不应直接被误判为医学核心术语。

## 6. 人工阅读建议

请人工打开 `reports/samples/sample_short_medium_long_for_review_limit500.md`，重点观察短、中、长样本的信息密度、摘要结构、缩写密度和是否适合整体入库。

## 7. 对 RAG 的影响

领域语言特点会影响 prompt 设计、query 改写、embedding 模型选择、metadata 保留和 answer citation。后续回答应尽量保留 PMID/PMCID/source_file，以支持追溯。
