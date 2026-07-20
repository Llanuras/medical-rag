# 全文切分策略分析说明

## 1. 本阶段分析目标

本阶段专门针对全文 `title + abstract + body` 制定切分策略。此前摘要级 Chroma 测试只使用 `title + abstract`，但全文 RAG 必须处理正文长度、章节结构和 chunk 数量增长问题。

## 2. 当前 `3028` 篇全文长度结论

- full text p95：`13091.25` tokens
- full text p99：`18109.17` tokens
- full text 超过 512 tokens 比例：`99.0753%`
- 按 `chunk_size=400, chunk_overlap=80` 估算全文总 chunks：`57717`

这些结果说明全文不能整体送入 embedding 模型，必须切分。

## 3. XML 正文章节结构

- 检测到正文 section title 的文献比例：`89.0026%`
- 章节标题统计见 `artifacts/metrics/t002_corpus_analysis/full_text_section_title_top50_limit3028.csv`
- 每篇文章的章节标记见 `artifacts/metrics/t002_corpus_analysis/full_text_section_analysis_limit3028.csv`

如果正文有明确章节标题，优先保留章节 metadata；如果章节标题缺失或结构不统一，再退回到递归滑动窗口。

## 4. 可选全文切分策略

- 全文整体不切分：不适合当前数据，因为绝大多数全文超过 512 tokens。
- 全量统一滑动窗口：实现简单，能稳定控制输入长度；缺点是可能切断章节语义。
- 章节优先 + 递归切分：先按 XML `sec/title` 保留章节，再对超长章节使用 recursive split；优点是更符合医学论文结构，缺点是实现更复杂。
- 仅摘要入库，正文作为补充：实现成本最低，但不能回答正文细节问题。

## 5. 推荐策略

当前推荐：正式全文 RAG 使用“章节优先 + RecursiveCharacterTextSplitter”的混合策略。

- 文本来源：`title + abstract + section_title + section_text`
- chunk 参数：`chunk_size=400, chunk_overlap=80`
- metadata：`record_id, source_file, title, journal, pub_year, pmid, pmcid, article_type, section_title, chunk_index, chunk_count`
- 对没有章节标题的正文：直接对全文 body 做 recursive split
- 对摘要级检索：继续保留 `title + abstract` 独立入库或作为高优先级字段

## 6. 对后续 RAG 的影响

全文入库会显著增加 chunk 数和 embedding 成本，但能覆盖摘要没有写出的实验细节、方法、结果和讨论。建议先做 50/100 篇全文 Chroma 小测试，再扩展到 `3028` 篇全文。
