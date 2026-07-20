# MeSH查询理解与增强模块说明（mesh）

## 1. T010 任务目标

T010 使用 MeSH XML 构建本地标准术语词典，并升级 T009 静态规则查询理解。未重新解析 PMC XML、未重建 embedding/Chroma，未调用 OpenAI，也未接入 UMLS。

## 2. 为什么先选 MeSH

MeSH 是 NLM 用于 PubMed/MEDLINE 索引的医学主题词表，与本项目 PMC/PubMed 检索场景直接匹配；其 XML 可公开下载并流式解析。UMLS 作为后续增强，不阻塞本轮。

## 3. XML 来源和解析范围

使用 NLM 2026 Production Year MeSH 的 Descriptor 与 Supplemental Concept XML.GZ。解析 DescriptorUI/Name/TreeNumber、Concept/Term/ScopeNote，以及 SupplementalRecord 与 HeadingMappedTo 信息；Qualifier 未提供时不伪造。

## 4. 词典构建与统计

- concept_count: 355148
- total_terms: 983486
- unique_terms: 983486
- descriptor_count: 31110
- supplemental_count: 324038
- terminology: /root/autodl-tmp/medical_rag/artifacts/terminology/mesh_2026/medical_synonyms_mesh.jsonl
- term_index: /root/autodl-tmp/medical_rag/artifacts/terminology/mesh_2026/term_to_concept_mesh.json

## 5. MeSH 优先查询理解

query_understanding.py 加载 JSONL concept groups 和 term index，英文术语按最长 n-gram 精确匹配，优先输出 concept_id、MeSH source、record type、tree_numbers 和同 concept 同义词。未命中的中文及项目常用词继续走静态 seed fallback，例如二甲双胍、心血管疾病、MI、PCR。

## 6. 清洗、扩展与检索 query

保留 T009 的空白/标点清洗、空 query 安全拦截和超长截断。每个 MeSH concept 最多保留五个短且去重的同义词；vector_query 保留原始核心词及增强词；bge_query 仍使用固定前缀 Represent this question for searching relevant passages: 。文档 embedding 不加该前缀。

## 7. Keyword 与 metadata filter

keyword_query 保持 required_terms、optional_terms 和 drug/disease/gene_protein/method/outcome buckets。MeSH record type 不能可靠映射细粒度实体类时保留在实体字段中，同时作为 disease bucket 的保守检索核心词。年份精确过滤仍为字符串 pub_year；范围、章节和不确定 article_type 继续进入 filter_plan。

## 8. 测试覆盖情况

共 15 条，MeSH 命中 query 12 条；ok=13，warning=1，invalid=1。覆盖 MI/myocardial infarction/heart attack、diabetes、breast/lung cancer、SARS、PCR、中文 fallback、年份、期刊、空和超长 query。

| raw_query | entities | sources | status |
| --- | ---: | --- | --- |
| 二甲双胍对心血管疾病有何影响？ | 2 | ["fallback_seed"] | ok |
| MI treatment | 2 | ["MeSH", "fallback_seed"] | ok |
| myocardial infarction treatment | 2 | ["MeSH"] | ok |
| heart attack aspirin mortality | 3 | ["MeSH"] | ok |
| EGFR mutation lung cancer treatment | 4 | ["MeSH", "fallback_seed"] | ok |
| HIV reverse transcriptase inhibitor resistance | 2 | ["MeSH", "fallback_seed"] | ok |
| type 2 diabetes insulin sensitivity after 2010 | 2 | ["MeSH"] | ok |
| PLoS ONE breast cancer gene expression | 2 | ["MeSH"] | ok |
| SARS coronavirus spike protein | 2 | ["MeSH"] | ok |
| PCR DNA amplification | 2 | ["MeSH"] | ok |
| warfarin bleeding risk | 3 | ["MeSH"] | ok |
| 2010年关于肺癌EGFR突变的研究 | 2 | ["fallback_seed"] | ok |
| metformin cardiovascular outcome in PLoS ONE | 2 | ["MeSH", "fallback_seed"] | ok |
| (empty) | 0 | [] | invalid |
| EGFR mutation EGFR mutation EGFR mutation EGFR mutation EGFR mutation EG | 2 | ["MeSH", "fallback_seed"] | warning |


## 9. 当前局限与下一步

MeSH 主要为英文主题词，中文映射仍是项目 seed；n-gram 字典匹配没有上下文消歧和拼写纠正；MeSH record type 不等于完整临床 NER 类型。下一步接入 Chroma 增强检索，并逐步增加 UMLS、hybrid retrieval、reranker 与 query rewrite。
