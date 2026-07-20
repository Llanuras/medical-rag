# 向量索引质量验证（limit153121_bge_base）

## 1. 基础统计

- Collection: `pmc_fulltext_bge_base_limit153121`
- Persist dir: `/root/autodl-tmp/medical_rag/artifacts/indexes/chroma/pmc_fulltext_bge_base_limit153121`
- Collection count: `3057078`
- Metadata complete in sample: `True`
- Self top1 hit rate: `0.95`
- Self top3 same-doc hit rate: `1.0`
- Elapsed: `31.2s`

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
| EGFR mutation lung cancer treatment | 2 | 0.2115715742111206 | PMCID:PMC2673583::chunk_00021 | PMCID:PMC2673583 | Customized Treatment in Non-Small-Cell Lung Cancer Based on EGFR Mutations and BRCA1 mRNA Expression | 2009 | Discussion |
| EGFR mutation lung cancer treatment | 3 | 0.2131490707397461 | PMCID:PMC2673583::chunk_00009 | PMCID:PMC2673583 | Customized Treatment in Non-Small-Cell Lung Cancer Based on EGFR Mutations and BRCA1 mRNA Expression | 2009 | Results |
| EGFR mutation lung cancer treatment | 4 | 0.21380996704101562 | PMCID:PMC1240071::chunk_00000 | PMCID:PMC1240071 | EGFR Mutations and Lung Cancer | 2005 | whole_document |
| EGFR mutation lung cancer treatment | 5 | 0.2138785719871521 | PMCID:PMC2673583::chunk_00000 | PMCID:PMC2673583 | Customized Treatment in Non-Small-Cell Lung Cancer Based on EGFR Mutations and BRCA1 mRNA Expression | 2009 | Introduction |
| HIV reverse transcriptase inhibitor resistance | 1 | 0.21514904499053955 | PMCID:PMC2770537::chunk_00015 | PMCID:PMC2770537 | Five-year follow up of genotypic resistance patterns in HIV-1 subtype C infected patients in Botswana after failure of thymidine analogue-based regimens | 2009 | Results |
| HIV reverse transcriptase inhibitor resistance | 2 | 0.21685725450515747 | PMCID:PMC2713244::chunk_00008 | PMCID:PMC2713244 | Virological efficacy and emergence of drug resistance in adults on antiretroviral treatment in rural Tanzania | 2009 | Results |
| HIV reverse transcriptase inhibitor resistance | 3 | 0.21747827529907227 | PMCID:PMC2951346::chunk_00013 | PMCID:PMC2951346 | Prevalence of Transmitted Drug Resistance and Impact of Transmitted Resistance on Treatment Success in the German HIV-1 Seroconverter Cohort | 2010 | Results |
| HIV reverse transcriptase inhibitor resistance | 4 | 0.22305744886398315 | PMCID:PMC2265747::chunk_00005 | PMCID:PMC2265747 | The incidence of multidrug and full class resistance in HIV-1 infected patients is decreasing over time (2001–2006) in Portugal | 2008 | Patients and Methods |
| HIV reverse transcriptase inhibitor resistance | 5 | 0.22353148460388184 | PMCID:PMC2100143::chunk_00005 | PMCID:PMC2100143 | N348I in the Connection Domain of HIV-1 Reverse Transcriptase Confers Zidovudine and Nevirapine Resistance | 2007 | Methods |
| type 2 diabetes insulin sensitivity | 1 | 0.2137993574142456 | PMCID:PMC2699827::chunk_00002 | PMCID:PMC2699827 | Type 2 Diabetes Mellitus: New Genetic Insights will Lead to New Therapeutics | 2009 | GLUCOSE HOMEOSTASIS AND DIABETES |
| type 2 diabetes insulin sensitivity | 2 | 0.2248726487159729 | PMCID:PMC2858103::chunk_00000 | PMCID:PMC2858103 | Advantages of the single delay model for the assessment of insulin sensitivity from the intravenous glucose tolerance test | 2010 | Background |
| type 2 diabetes insulin sensitivity | 3 | 0.22905850410461426 | PMCID:PMC1309619::chunk_00014 | PMCID:PMC1309619 | Prevalence, predisposition and prevention of type II diabetes | 2005 | Obesity, Metabolic Syndrome, prediabetes |
| type 2 diabetes insulin sensitivity | 4 | 0.23323404788970947 | PMCID:PMC2694824::chunk_00018 | PMCID:PMC2694824 | Indices of insulin sensitivity and secretion from a standard liquid meal test in subjects with type 2 diabetes, impaired or normal fasting glucose | 2009 | Results |
| type 2 diabetes insulin sensitivity | 5 | 0.2353118658065796 | PMCID:PMC2271045::chunk_00006 | PMCID:PMC2271045 | A Survey of Insulin-Dependent Diabetes—Part I: Therapies and Devices | 2008 | 2. DIABETES MELLITUS |
| SARS coronavirus spike protein | 1 | 0.1938180923461914 | PMCID:PMC549520::chunk_00011 | PMCID:PMC549520 | The Severe Acute Respiratory Syndrome (SARS)-coronavirus 3a protein may function as a modulator of the trafficking properties of the spike protein | 2005 | Implication of the hypothesis |
| SARS coronavirus spike protein | 2 | 0.20161467790603638 | PMCID:PMC2805634::chunk_00021 | PMCID:PMC2805634 | Mutagenesis of the transmembrane domain of the SARS coronavirus spike glycoprotein: refinement of the requirements for SARS coronavirus cell entry | 2009 | Methods |
| SARS coronavirus spike protein | 3 | 0.2063843011856079 | PMCID:PMC548145::chunk_00011 | PMCID:PMC548145 | Molecular mechanisms of severe acute respiratory syndrome (SARS) | 2005 | Introduction |
| SARS coronavirus spike protein | 4 | 0.2072938084602356 | PMCID:PMC2805634::chunk_00002 | PMCID:PMC2805634 | Mutagenesis of the transmembrane domain of the SARS coronavirus spike glycoprotein: refinement of the requirements for SARS coronavirus cell entry | 2009 | Background |
| SARS coronavirus spike protein | 5 | 0.21041840314865112 | PMCID:PMC549520::chunk_00009 | PMCID:PMC549520 | The Severe Acute Respiratory Syndrome (SARS)-coronavirus 3a protein may function as a modulator of the trafficking properties of the spike protein | 2005 | Implication of the hypothesis |

## 4. Metadata Filter 验证

| filter_name | where | hit_count | status | note |
| --- | --- | --- | --- | --- |
| split_strategy_semantic_section | {'split_strategy': 'semantic_section'} | 5 | ok | matched |
| article_type_research_article | {'article_type': 'research-article'} | 5 | ok | matched |
| pub_year_2010_string | {'pub_year': '2010'} | 5 | ok | matched |
| journal_plos_one | {'journal': 'PLoS ONE'} | 5 | ok | matched |
| sample_pub_year | {'pub_year': '2003'} | 5 | ok | matched |
| real_pmcid | {'pmcid': 'PMC176545'} | 5 | ok | matched |
| no_result_filter | {'pmcid': '__NO_SUCH_PMCID__'} | 0 | ok | empty result expected |

## 5. 边界 Query

- 空查询：已在脚本内拦截，不调用模型。
- 超长查询：脚本会压缩空白并截断到 4000 字符再检索。
- 无结果 filter：返回空结果，不视为失败。
