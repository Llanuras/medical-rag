# PMC XML 解析策略说明

## 本阶段分析目标

本阶段将本地 PMC OA `oa_comm/xml` 文件解析为结构化 records，作为后续字段质量、token 长度、领域语言和 Chroma 入库分析的统一数据基础。

## 解析方式

- 数据来源：`data/raw/pmc_oa_comm`
- 目标处理数量：`3028`
- 实际选择 XML：`3028`
- 解析成功：`3028`
- 解析失败：`0`

脚本使用本地 XML parser 提取字段，并通过 `datasets.Dataset.from_list(records)` 构建本地 HuggingFace Dataset 对象，体现 data pipeline。该流程不会从 HuggingFace 下载其他医学数据集。

## 与直接 load_dataset 在线数据集相比

使用本地 XML 的优点是严格符合 PMC OA `oa_comm` 数据源要求，可保留 source_file、pmid、pmcid 等溯源字段，并能观察真实 XML 结构差异。缺点是需要自行处理字段缺失、标签嵌套、正文过长和结构不统一问题。

直接使用在线 `load_dataset` 的优点是更方便、结构更统一；缺点是可能不符合本任务指定数据源，且难以确认样本确实来自当前本地 PMC OA `oa_comm` XML。

## 核心字段选择原因

- `title`：可增强检索文本，帮助短摘要样本提供主题信息。
- `abstract`：初始 RAG 最核心文本，信息密度高且长度通常可控。
- `body`：后续全文 RAG 的候选文本，但长度较长，需要切分策略。
- `journal`：未来可作为期刊 metadata filter。
- `pub_date/pub_year`：未来可作为时间过滤器，例如近 5 年文献检索。
- `pmid/pmcid`：用于 PubMed/PMC 原文追溯和回答 citation/source tracking。
- `source_file`：用于调试解析问题和本地数据溯源。

## XML 解析风险

PMC XML 可能存在字段缺失、结构标签不统一、结构化摘要嵌套复杂、正文过长、部分文献缺少 PMID 或期刊信息等问题。因此后续所有分析必须基于真实解析结果，不能写死或编造结论。
