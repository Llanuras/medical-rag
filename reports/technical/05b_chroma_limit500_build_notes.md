# 500 篇 Chroma 测试库构建说明

## 本阶段目标

在最小 embedding/Chroma 测试通过后，本阶段构建 500 篇样本范围内的 Chroma 测试库。该库用于验证批量入库流程，不等同于最终生产级 RAG 知识库。

## 入库策略

- 入库文本：`title + abstract`
- 入库记录：`quality_decision` 为 `keep` 或 `keep_with_warning`，且 `text_title_abstract` 非空
- metadata：`record_id, source_file, title, journal, pub_year, pmid, pmcid, article_type`
- 分割策略：`title+abstract <= 512 tokens` 整体入库；超过 512 tokens 的长尾样本使用 `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)`。

## 当前结果

- 输入记录数：`500`
- 实际入库记录数：`500`
- 实际写入 chunk 数：`731`
- 持久化目录：`archive/experiments/indexes/chroma_limit500`

## 对后续 RAG 的影响

该库证明 500 篇摘要级数据可以进入 Chroma，并保留 metadata 供后续过滤和溯源。正式 RAG 构建时可扩展到更多文献，并根据全文 token 分析决定是否加入 body 和章节切分。
