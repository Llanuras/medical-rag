# MeSH标准术语词典构建报告（mesh）

## 1. 数据来源与范围

标准术语来源为 NLM 当前生产年度 MeSH XML：Descriptor Records 与 Supplemental Concept Records；本轮未接入 UMLS。输入为压缩 XML，采用 xml.etree.ElementTree.iterparse 流式处理，不复制 XML 原文到 outputs。

## 2. 构建方式

每个 Descriptor/SupplementalRecord 聚合 preferred term、ConceptName 与 TermList 为一个同义词组。术语 key 统一为去首尾空白、压缩空白、casefold 后的值；过滤短于最小长度、纯数字和超过 256 字符的噪声词，每个 concept 最多保留指定数量的术语。

## 3. 统计

- descriptor_count: 31110
- supplemental_count: 324038
- qualifier_count: 0
- concept_count: 355148
- total_terms: 983486
- unique_terms: 983486
- skipped_terms: 12

## 4. 输出

- synonym JSONL: artifacts/terminology/mesh_2026/medical_synonyms_mesh.jsonl
- synonym CSV: artifacts/terminology/mesh_2026/medical_synonyms_mesh.csv
- term index: artifacts/terminology/mesh_2026/term_to_concept_mesh.json
- stats JSON: artifacts/metrics/t010_mesh_query_understanding/medical_terminology_stats_mesh.json

## 5. 使用说明与局限

term_to_concept_mesh.json 用于最长匹配和 concept 定位；medical_synonyms_mesh.jsonl 用于从 concept 取得同义词组。MeSH 是英文标准主题词表，中文医学词仍由项目 seed fallback 处理。Supplemental Record 的 tree_numbers 字段保存其 HeadingMappedTo DescriptorUI；这不是 Descriptor Tree Number。
