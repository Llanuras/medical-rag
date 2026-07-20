# 医学全文领域内容理解报告

## 1. 任务口径

本报告按全文 `title + abstract + body` 完成领域内容理解，而不是只看摘要。分层依据为 `full_token_len`，每层抽样 `8` 篇，共 `24` 篇全文；完整抽样全文保存在 `reports/samples/fulltext_stratified_sample_limit3028.jsonl`，人工阅读版保存在 `reports/samples/fulltext_stratified_sample_for_review_limit3028.md`。

## 2. 全文分层抽样

- short：`<= 4130` tokens
- medium：`4132-7101` tokens
- long：`>= 7103` tokens

抽样明细见 `artifacts/metrics/t002_corpus_analysis/fulltext_domain_sample_summary_limit3028.csv`。短文多为评论、社论、通信、简短病例或无标准正文结构的文章；中长文更常出现结构化章节和方法/结果细节。

## 3. 结构：是否遵循 IMRaD

在 3028 篇全文中，检测到任意正文 section title 的比例为 `89.0%`；检测到 Introduction/Background + Methods + Results + Discussion 的 IMRaD core 比例为 `67.1%`；如果要求额外包含 Conclusion/Summary，则比例为 `52.1%`。

在本次 `24` 篇全文样本中，IMRaD core 比例为 `70.8%`，包含 Conclusion/Summary 的比例为 `62.5%`。结论是：研究型原始论文较稳定遵循 IMRaD，但短文、社论、评论、软件说明、病例报告和部分 BMC 早期文章会使用 Background/Discussion/Summary、Case presentation、Availability and requirements 等非标准标题。后续 prompt 和评估规则不应硬编码只识别 Introduction/Methods/Results/Discussion 四个英文标题。

## 4. 术语与缩写密度

全文 top 缩写包括：`DNA, RNA, PCR, II, HIV, CI, SD, CA, PBS, HIV-1, GFP, CD4`。这些缩写覆盖分子生物学、感染病、统计学和临床研究场景，例如 DNA/RNA/PCR/IL/IFN/TNF 更偏基础与免疫机制，HIV/AIDS/CSF/PCP 偏疾病或临床对象，CI/OR/RR/SD 偏统计表达。

本语料明显存在“缩写 + 全称 + 同义词/近义词”并存的问题。样本中可见多种表述的概念包括：`HIV/AIDS, PCR, confidence interval, odds ratio, corticosteroids`。全集概念变体统计见 `artifacts/metrics/t002_corpus_analysis/fulltext_concept_variants_limit3028.csv`。后续 query 改写需要保留原词，同时扩展常见全称和缩写，例如 HIV/AIDS/human immunodeficiency virus、PCR/polymerase chain reaction、CSF/cerebrospinal fluid、PCP/Pneumocystis pneumonia。

## 5. 常用专业术语清单

- 分子/基因/细胞：DNA, RNA, PCR, RT-PCR, gene expression, amino acid, cell line, protein, transcription factor, SNP, GFP, BLAST, ATP。
- 免疫/感染/病毒：HIV, HIV-1, AIDS, IFN, TNF, IL, CD4, CD8, HSV, LPS, Pneumocystis pneumonia/PCP。
- 临床/疾病/患者：patients, treatment, control group, risk factors, breast cancer, prostate cancer, COPD, asthma, depression, psoriasis, myocardial infarction/MI。
- 统计/流行病学：CI, 95% CI, OR, RR, SD, ANOVA, statistically significant, sample size, standard deviation, prevalence, mortality。
- 公共卫生/医疗服务：health care, public health, United States, quality of life, data set, data analysis。

这些术语不应只按单词匹配处理。尤其是疾病、检测技术和统计效应量常以缩写、全称、连字符写法和复数形式同时出现。

## 6. 高频词与专业语言风格

全文高频单词 top 项包括：`cells, data, genes, patients, expression, gene, cell, analysis, protein, different, health, time`。高频短语 top 项包括：`gene expression, united states, amino acid, cell lines, health care, statistically significant, amino acids, statistical analysis, breast cancer, patients who`。

这些词说明该 3028 篇集合不是单一临床疾病语料，而是混合了临床研究、基础生物学、微生物/病毒、公共卫生、遗传与蛋白表达分析。医学文本信息密度高，常把研究对象、干预/暴露、检测技术、统计效应量、样本来源、时间和条件压缩在一句话中；一句话内常同时出现数字、括号、缩写、基因/蛋白名和统计符号。

本次全文样本的信息密度指标如下：

| group | n | token range | avg sentence words | avg abbr / 1000 words | IMRaD core |
|---|---:|---:|---:|---:|---:|
| short | 8 | 344-4091 | 17.6 | 26.5 | 37.5% |
| medium | 8 | 4168-6839 | 17.9 | 41.3 | 87.5% |
| long | 8 | 7262-11630 | 17.6 | 35.9 | 87.5% |

## 7. 对提示词工程的启发

- 检索 query 改写：对缩写做双向扩展，但回答时保留原文术语和 PMCID/PMID。
- 结构化抽取：章节识别应支持 Background/Introduction、Materials and methods/Patients and methods、Results and discussion、Conclusion/Summary、Case presentation 等变体。
- 答案生成：要求模型区分研究背景、方法、主要结果、限制和结论，避免把 Discussion 中的推测写成实证结果。
- 评估基线：不要只用通顺度评估，应检查是否保留疾病/基因/药物/统计缩写、数值和研究对象限定条件。
- 全文 RAG：长文必须按章节优先切分；短文可以整体或低粒度切分，但仍需保留 article_type 和 section metadata。

## 8. 输出文件

- `artifacts/metrics/t002_corpus_analysis/fulltext_domain_sample_summary_limit3028.csv`
- `reports/samples/fulltext_stratified_sample_for_review_limit3028.md`
- `reports/samples/fulltext_stratified_sample_limit3028.jsonl`
- `artifacts/metrics/t002_corpus_analysis/fulltext_abbreviation_top50_limit3028.csv`
- `artifacts/metrics/t002_corpus_analysis/fulltext_high_freq_unigrams_limit3028.csv`
- `artifacts/metrics/t002_corpus_analysis/fulltext_high_freq_bigrams_limit3028.csv`
- `artifacts/metrics/t002_corpus_analysis/fulltext_high_freq_trigrams_limit3028.csv`
- `artifacts/metrics/t002_corpus_analysis/fulltext_concept_variants_limit3028.csv`
