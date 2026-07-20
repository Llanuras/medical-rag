# 向量索引质量验证（part000_limit153121_bge_base）

## 1. 基础统计

- Collection: `pmc_fulltext_bge_base_part000_limit153121`
- Persist dir: `/root/autodl-tmp/medical_rag/artifacts/indexes/chroma/pmc_fulltext_bge_base_part000_limit153121`
- Collection count: `64853`
- Metadata complete in sample: `True`
- Self top1 hit rate: `0.95`
- Self top3 same-doc hit rate: `1.0`
- Elapsed: `7.0s`

## 2. 自相似性验证

| sample_id | chunk_id | doc_id | top1_chunk_id | top1_is_self | top3_same_doc |
| --- | --- | --- | --- | --- | --- |
| 0 | PMCID:PMC176545::chunk_00000 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00000 | True | True |
| 1 | PMCID:PMC176545::chunk_00001 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00001 | True | True |
| 2 | PMCID:PMC176545::chunk_00002 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00002 | True | True |
| 3 | PMCID:PMC176545::chunk_00003 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00003 | True | True |
| 4 | PMCID:PMC176545::chunk_00004 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00004 | True | True |
| 5 | PMCID:PMC176545::chunk_00005 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00005 | True | True |
| 6 | PMCID:PMC176545::chunk_00006 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00006 | True | True |
| 7 | PMCID:PMC176545::chunk_00007 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00007 | True | True |
| 8 | PMCID:PMC176545::chunk_00008 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00008 | True | True |
| 9 | PMCID:PMC176545::chunk_00009 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00009 | True | True |
| 10 | PMCID:PMC176545::chunk_00010 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00010 | True | True |
| 11 | PMCID:PMC176545::chunk_00011 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00010 | False | True |
| 12 | PMCID:PMC176545::chunk_00012 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00012 | True | True |
| 13 | PMCID:PMC176545::chunk_00013 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00013 | True | True |
| 14 | PMCID:PMC176545::chunk_00014 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00014 | True | True |
| 15 | PMCID:PMC176545::chunk_00015 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00015 | True | True |
| 16 | PMCID:PMC176545::chunk_00016 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00016 | True | True |
| 17 | PMCID:PMC176545::chunk_00017 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00017 | True | True |
| 18 | PMCID:PMC176545::chunk_00018 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00018 | True | True |
| 19 | PMCID:PMC176545::chunk_00019 | PMCID:PMC176545 | PMCID:PMC176545::chunk_00019 | True | True |

## 3. 医学 Query 检索

| query | rank | distance | chunk_id | doc_id | source_title | pub_year | section_title |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EGFR mutation lung cancer treatment | 1 | 0.21027427911758423 | PMCID:PMC545205::chunk_00000 | PMCID:PMC545205 | Gene Mutations in Lung Cancer: Promising Predictive Factors for the Success of Molecular Therapy | 2005 | Molecular Therapy for Lung Cancer |
| EGFR mutation lung cancer treatment | 2 | 0.22255825996398926 | PMCID:PMC545205::chunk_00003 | PMCID:PMC545205 | Gene Mutations in Lung Cancer: Promising Predictive Factors for the Success of Molecular Therapy | 2005 | Therapeutic Implications |
| EGFR mutation lung cancer treatment | 3 | 0.22613996267318726 | PMCID:PMC545205::chunk_00001 | PMCID:PMC545205 | Gene Mutations in Lung Cancer: Promising Predictive Factors for the Success of Molecular Therapy | 2005 | EGFR Gene Mutations in NSCLC |
| EGFR mutation lung cancer treatment | 4 | 0.24778234958648682 | PMCID:PMC545207::chunk_00010 | PMCID:PMC545207 | KRAS Mutations and Primary Resistance of Lung Adenocarcinomas to Gefitinib or Erlotinib | 2005 | Discussion |
| EGFR mutation lung cancer treatment | 5 | 0.25183796882629395 | PMCID:PMC516034::chunk_00000 | PMCID:PMC516034 | Evaluation of safety and efficacy of gefitinib ('iressa', zd1839) as monotherapy in a series of Chinese patients with advanced non-small-cell lung cancer: experience from a compassionate-use programme | 2004 | Background |
| HIV reverse transcriptase inhibitor resistance | 1 | 0.2539515495300293 | PMCID:PMC539050::chunk_00013 | PMCID:PMC539050 | Randomized, Controlled Trial of Therapy Interruption in Chronic HIV-1 Infection | 2004 | Results |
| HIV reverse transcriptase inhibitor resistance | 2 | 0.27863818407058716 | PMCID:PMC554760::chunk_00002 | PMCID:PMC554760 | New York City HIV superbug: fear or fear not? | 2005 | recursive_fallback |
| HIV reverse transcriptase inhibitor resistance | 3 | 0.28423309326171875 | PMCID:PMC554760::chunk_00003 | PMCID:PMC554760 | New York City HIV superbug: fear or fear not? | 2005 | recursive_fallback |
| HIV reverse transcriptase inhibitor resistance | 4 | 0.28561270236968994 | PMCID:PMC539050::chunk_00014 | PMCID:PMC539050 | Randomized, Controlled Trial of Therapy Interruption in Chronic HIV-1 Infection | 2004 | Results |
| HIV reverse transcriptase inhibitor resistance | 5 | 0.28724050521850586 | PMCID:PMC539050::chunk_00019 | PMCID:PMC539050 | Randomized, Controlled Trial of Therapy Interruption in Chronic HIV-1 Infection | 2004 | Discussion |
| type 2 diabetes insulin sensitivity | 1 | 0.26487934589385986 | PMCID:PMC546416::chunk_00000 | PMCID:PMC546416 | A case study of type 2 diabetes self-management | 2005 | Background |
| type 2 diabetes insulin sensitivity | 2 | 0.269905149936676 | PMCID:PMC544854::chunk_00001 | PMCID:PMC544854 | Transcriptional regulation of lipid metabolism by fatty acids: a key determinant of pancreatic β-cell function | 2005 | Type 2 diabetes and free fatty acids |
| type 2 diabetes insulin sensitivity | 3 | 0.2735356092453003 | PMCID:PMC524031::chunk_00006 | PMCID:PMC524031 | Metabolic response of people with type 2 diabetes to a high protein diet | 2004 | Results |
| type 2 diabetes insulin sensitivity | 4 | 0.27485668659210205 | PMCID:PMC546416::chunk_00002 | PMCID:PMC546416 | A case study of type 2 diabetes self-management | 2005 | Background |
| type 2 diabetes insulin sensitivity | 5 | 0.2814415693283081 | PMCID:PMC549588::chunk_00009 | PMCID:PMC549588 | Case-Based Study: From Prediabetes to Complications—Opportunities for Prevention | 2005 | DISCUSSION |
| SARS coronavirus spike protein | 1 | 0.1938180923461914 | PMCID:PMC549520::chunk_00011 | PMCID:PMC549520 | The Severe Acute Respiratory Syndrome (SARS)-coronavirus 3a protein may function as a modulator of the trafficking properties of the spike protein | 2005 | Implication of the hypothesis |
| SARS coronavirus spike protein | 2 | 0.2063843011856079 | PMCID:PMC548145::chunk_00011 | PMCID:PMC548145 | Molecular mechanisms of severe acute respiratory syndrome (SARS) | 2005 | Introduction |
| SARS coronavirus spike protein | 3 | 0.21041840314865112 | PMCID:PMC549520::chunk_00009 | PMCID:PMC549520 | The Severe Acute Respiratory Syndrome (SARS)-coronavirus 3a protein may function as a modulator of the trafficking properties of the spike protein | 2005 | Implication of the hypothesis |
| SARS coronavirus spike protein | 4 | 0.21095454692840576 | PMCID:PMC549520::chunk_00000 | PMCID:PMC549520 | The Severe Acute Respiratory Syndrome (SARS)-coronavirus 3a protein may function as a modulator of the trafficking properties of the spike protein | 2005 | Background |
| SARS coronavirus spike protein | 5 | 0.21211063861846924 | PMCID:PMC549520::chunk_00008 | PMCID:PMC549520 | The Severe Acute Respiratory Syndrome (SARS)-coronavirus 3a protein may function as a modulator of the trafficking properties of the spike protein | 2005 | Implication of the hypothesis |

## 4. Metadata Filter 验证

| filter_name | where | hit_count | status | note |
| --- | --- | --- | --- | --- |
| split_strategy_semantic_section | {'split_strategy': 'semantic_section'} | 5 | ok | matched |
| article_type_research_article | {'article_type': 'research-article'} | 5 | ok | matched |
| pub_year_2010 | {'pub_year': 2010} | 0 | ok | no matching rows in this collection or test subset |
| journal_plos_one | {'journal': 'PLoS ONE'} | 0 | ok | no matching rows in this collection or test subset |
| real_pmcid | {'pmcid': 'PMC176545'} | 5 | ok | matched |
| no_result_filter | {'pmcid': '__NO_SUCH_PMCID__'} | 0 | ok | empty result expected |

## 5. 边界 Query

- 空查询：已在脚本内拦截，不调用模型。
- 超长查询：脚本会压缩空白并截断到 4000 字符再检索。
- 无结果 filter：返回空结果，不视为失败。
